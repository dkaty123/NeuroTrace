import inspect
import json
import os
import requests
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional, TypeVar, Generic, Type, get_type_hints, List
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

T = TypeVar('T')

class SpectraCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to capture LLM thinking and processing."""
    
    def __init__(self, spectra_instance):
        super().__init__()
        self.spectra = spectra_instance
        self.current_node = None
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts processing."""
        for i, prompt in enumerate(prompts):
            self.spectra._log_to_jsonl({
                "event": "llm_thinking_start",
                "timestamp": datetime.now().isoformat(),
                "current_node": self.current_node,
                "prompt": prompt,
                "llm_params": serialized,
                "prompt_index": i
            })
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes processing."""
        for i, generation in enumerate(response.generations):
            for j, gen in enumerate(generation):
                self.spectra._log_to_jsonl({
                    "event": "llm_thinking_end",
                    "timestamp": datetime.now().isoformat(),
                    "current_node": self.current_node,
                    "response": gen.text,
                    "generation_info": gen.generation_info,
                    "prompt_index": i,
                    "generation_index": j
                })
        
        # Log token usage if available
        if hasattr(response, 'llm_output') and response.llm_output:
            self.spectra._log_to_jsonl({
                "event": "llm_token_usage",
                "timestamp": datetime.now().isoformat(),
                "current_node": self.current_node,
                "token_usage": response.llm_output.get('token_usage', {}),
                "model_name": response.llm_output.get('model_name', 'unknown')
            })
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts."""
        chain_type = 'unknown'
        if serialized and isinstance(serialized, dict):
            chain_type = serialized.get('name', serialized.get('id', 'unknown'))
        
        self.spectra._log_to_jsonl({
            "event": "chain_thinking_start",
            "timestamp": datetime.now().isoformat(),
            "current_node": self.current_node,
            "chain_type": chain_type,
            "inputs": inputs if inputs else {}
        })

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain ends."""
        self.spectra._log_to_jsonl({
            "event": "chain_thinking_end",
            "timestamp": datetime.now().isoformat(),
            "current_node": self.current_node,
            "outputs": outputs
        })

class Spectra:
    def __init__(self, workflow: Any, output_dir: str = "logs", live_logging: bool = True, capture_thinking: bool = True, 
                 auto_process: bool = True, api_url: str = "http://localhost:3000/api/process-logs", 
                 agent_code_api_url: str = "http://localhost:3000/api/process-agent-code", api_cooldown: float = 0.5):
        """
        Initialize the Spectra logger for LangGraph workflows.
        
        Args:
            workflow: The LangGraph workflow to analyze
            output_dir: Directory to store the log files (relative to current working directory)
            live_logging: Whether to enable live JSONL logging during execution
            capture_thinking: Whether to capture LLM thinking (prompts, responses, chains)
            auto_process: Whether to automatically call the process-logs API when new logs are written
            api_url: URL of the process-logs API endpoint
            agent_code_api_url: URL of the process-agent-code API endpoint
            api_cooldown: Minimum seconds between API calls (default 0.5)
        """
        self.workflow = workflow
        self.output_dir = output_dir
        self.live_logging = live_logging
        self.capture_thinking = capture_thinking
        self.auto_process = auto_process
        self.api_url = api_url
        self.agent_code_api_url = agent_code_api_url
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.chain = None
        self.pending_logs = []  # Buffer for logs to be sent to API
        self.last_api_call = 0  # Timestamp of last API call
        self.api_cooldown = api_cooldown  # Minimum seconds between API calls
        self.log_buffer_lock = threading.Lock()  # Initialize the lock
        self._api_call_timer = None  # To keep track of the timer
        
        # Clear any existing processed data first for this new workflow instance
        self._clear_processed_data()
        
        # Initialize callback handler for LLM thinking
        if self.capture_thinking:
            self.callback_handler = SpectraCallbackHandler(self)
        
        # Get the state type from the workflow's nodes
        self.state_type = None
        for node_name, node_spec in workflow.nodes.items():
            node_func = self._get_function_from_spec(node_spec)
            if node_func is not None:
                type_hints = get_type_hints(node_func)
                if 'state' in type_hints:
                    self.state_type = type_hints['state']
                    break
        
        if self.state_type is None:
            # Try to infer from workflow.state_schema if available (LangGraph >= 0.0.40)
            if hasattr(workflow, 'state_schema') and workflow.state_schema:
                self.state_type = workflow.state_schema
            else:
                raise ValueError("Could not determine state type from workflow nodes or workflow.state_schema.")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize JSONL log file if live logging is enabled
        if self.live_logging:
            self.jsonl_file = f"{self.output_dir}/workflow_execution_{self.timestamp}.jsonl"
            # Initialize the file with metadata
            self._log_to_jsonl({
                "event": "workflow_initialized",
                "timestamp": datetime.now().isoformat(),
                "workflow_id": self.timestamp,
                "state_type": str(self.state_type.__name__ if hasattr(self.state_type, '__name__') else str(self.state_type)),
                "capture_thinking": self.capture_thinking
            })
        
        # Log the node functions
        self._log_node_functions()
        
        # Compile the workflow
        self.compile()
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        else:
            # For complex objects, convert to string representation
            return str(obj)
    
    def _log_to_jsonl(self, data: Dict[str, Any]):
        """Append a JSON line to the JSONL log file."""
        if self.live_logging:
            try:
                # Ensure data is serializable
                serializable_data = self._make_serializable(data)
                with open(self.jsonl_file, 'a') as f:
                    f.write(json.dumps(serializable_data) + '\n')
                    f.flush()  # Ensure immediate write
                
                # Add to pending logs for API processing if auto_process is enabled
                if self.auto_process:
                    with self.log_buffer_lock:  # Protect append
                        self.pending_logs.append(serializable_data)
                    self._schedule_api_call()
                    
            except Exception as e:
                # If serialization fails, log the error instead
                error_data = {
                    "event": "serialization_error",
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e),
                    "original_event_type": data.get("event", "unknown"),
                    "original_data_preview": str(data)[:200]  # Log a preview of the unserializable data
                }
                try:  # Try to write the error to the log file
                    with open(self.jsonl_file, 'a') as f:
                        f.write(json.dumps(self._make_serializable(error_data)) + '\n')
                        f.flush()
                except Exception as efs:  # If even error logging fails, print to console
                    print(f"CRITICAL: Failed to write serialization error to log file: {efs}. Original error: {e}")
    
    def _schedule_api_call(self):
        """Schedule an API call to process logs, with cooldown to batch requests."""
        with self.log_buffer_lock:  # Protect check on pending_logs
            if not self.pending_logs:
                return
            
        current_time = time.time()
        if current_time - self.last_api_call >= self.api_cooldown:
            self._call_process_logs_api()
        else:
            with self.log_buffer_lock:  # Protect timer management
                if self._api_call_timer is None or not self._api_call_timer.is_alive():
                    delay = max(0, self.api_cooldown - (current_time - self.last_api_call))
                    self._api_call_timer = threading.Timer(delay, self._call_process_logs_api)
                    # self._api_call_timer.daemon = False # Non-daemon thread
                    self._api_call_timer.start()
    
    def _call_process_logs_api(self):
        """Call the process-logs API with pending logs."""
        logs_to_send = []
        with self.log_buffer_lock:  # Protect read and clear of pending_logs
            if not self.pending_logs:
                return  # Nothing to send
            logs_to_send.extend(self.pending_logs)
            self.pending_logs.clear()
        
        if not logs_to_send:
            return

        self.last_api_call = time.time()  # Update last call time as we are initiating a send
        
        def make_request():
            try:
                response = requests.post(
                    self.api_url,
                    json=logs_to_send,
                    headers={'Content-Type': 'application/json'},
                    timeout=15  # Increased timeout slightly
                )
                if response.status_code == 200:
                    try:
                        result = response.json()
                        print(f"✅ Processed {len(logs_to_send)} logs via API. Server reports total: {result.get('summary', {}).get('total_entries_in_file', 'unknown')}")
                    except json.JSONDecodeError:
                        print(f"✅ Processed {len(logs_to_send)} logs via API (non-JSON response: {response.text[:100]})")
                else:
                    print(f"⚠️ API call failed with status {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Failed to call process-logs API: {e}")
            except Exception as e:
                print(f"⚠️ Unexpected error calling API: {e}")
            finally:
                with self.log_buffer_lock:  # Protect check on pending_logs
                    if self.pending_logs:
                        self._schedule_api_call()  # Re-schedule if new logs came in
            
        # Make threads non-daemon so main script waits for them
        api_thread = threading.Thread(target=make_request)  # daemon=False is default
        api_thread.start()
            
    def _clear_processed_data(self):
        """Send a DELETE request to clear processed agent code and logs."""
        try:
            # Use the host from the agent_code_api_url for consistency
            base_url = self.agent_code_api_url.split('/api/')[0]
            clear_url = f"{base_url}/api/process-agent-code"  # Corrected endpoint
            
            response = requests.delete(clear_url, timeout=10)
            if response.status_code == 200:
                print(f"✅ Successfully cleared processed data via DELETE to {clear_url}")
            else:
                print(f"⚠️ Failed to clear processed data via DELETE to {clear_url}. Status: {response.status_code}, Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ RequestException when trying to clear processed data: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error when trying to clear processed data: {e}")

    def _send_agent_code_to_api(self, agent_id: str, agent_name: str, source_code: str, 
                               description: str = "", file_path: str = "", dependencies: List[str] = None):
        """Send agent code data to the process-agent-code API endpoint."""
        if not self.auto_process:
            return
            
        try:
            agent_data = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "source_code": source_code,
                "description": description,
                "file_path": file_path,
                "dependencies": dependencies or [],
                "timestamp": datetime.now().isoformat()
            }
            
            def make_request():
                try:
                    response = requests.post(
                        self.agent_code_api_url,
                        json=agent_data,
                        headers={'Content-Type': 'application/json'},
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"✅ Sent agent code for {agent_name} ({agent_id}) to API")
                    else:
                        print(f"⚠️ Agent code API call failed with status {response.status_code}: {response.text}")
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Failed to call agent code API: {e}")
                except Exception as e:
                    print(f"⚠️ Unexpected error calling agent code API: {e}")
            
            # Run in background thread
            agent_thread = threading.Thread(target=make_request)  # daemon=False is default
            agent_thread.start()
            
        except Exception as e:
            print(f"⚠️ Error sending agent code to API: {e}")
    
    def _get_function_from_spec(self, spec):
        """Extract the original function from a StateNodeSpec."""
        if hasattr(spec, 'runnable'):
            runnable = spec.runnable
            # Check if runnable is a FunctionType or MethodType first for direct callables
            if inspect.isfunction(runnable) or inspect.ismethod(runnable):
                return runnable
            # Then check for .func for specific LangGraph structures
            if hasattr(runnable, 'func'):
                # Further check if runnable.func is itself a callable (e.g. functools.partial)
                if callable(runnable.func):
                    return runnable.func
                # If runnable.func is not directly callable, it might be a deeper structure
                # This part might need more specific handling if LangGraph changes its internal structure
        # Fallback or if spec is already a callable function/method
        if callable(spec):
            return spec
        return None
    
    def _log_node_functions(self):
        """Log the source code of all node functions in the workflow."""
        node_functions_log = {
            "workflow_id": self.timestamp,
            "state_type": str(self.state_type),
            "nodes": {}
        }
        
        # Get all nodes from the workflow
        for node_name, node_spec in self.workflow.nodes.items():
            # Extract the original function from the StateNodeSpec
            node_func = self._get_function_from_spec(node_spec)
            
            if node_func is not None:
                try:
                    # Get source code of the node function
                    source = inspect.getsource(node_func)
                    
                    # Get the function's type hints
                    signature = inspect.signature(node_func)
                    
                    # Get function metadata
                    metadata = {
                        "name": node_func.__name__,
                        "doc": node_func.__doc__ or "",
                        "module": node_func.__module__,
                        "input_type": str(signature.parameters.get('state', 'Unknown'))  # Try to get state param type
                    }
                    
                    node_functions_log["nodes"][node_name] = {
                        "source_code": source,
                        "return_type": str(signature.return_annotation),
                        "metadata": metadata,
                        "timestamp": self.timestamp
                    }
                    
                    # Send agent code to API
                    self._send_agent_code_to_api(
                        agent_id=f"{self.timestamp}_{node_name}",
                        agent_name=node_name,
                        source_code=source,
                        description=node_func.__doc__ or f"Agent function: {node_func.__name__}",
                        file_path=f"{node_func.__module__}",
                        dependencies=[]  # Could be enhanced to detect imports
                    )
                    
                    # Log to JSONL as well
                    if self.live_logging:
                        self._log_to_jsonl({
                            "event": "node_function_logged",
                            "timestamp": datetime.now().isoformat(),
                            "node_name": node_name,
                            "function_name": node_func.__name__,
                            "doc": node_func.__doc__ or "",
                            "module": node_func.__module__
                        })
                        
                except (TypeError, OSError, AttributeError) as e:  # Added AttributeError
                    err_msg = f"Could not get source/metadata for {node_name}: {str(e)}"
                    node_functions_log["nodes"][node_name] = {
                        "error": err_msg,
                        "timestamp": self.timestamp
                    }
                    
                    # Log error to JSONL
                    if self.live_logging:
                        self._log_to_jsonl({
                            "event": "node_function_error",
                            "timestamp": datetime.now().isoformat(),
                            "node_name": node_name,
                            "error": err_msg
                        })
        
        # Save to JSON file
        node_functions_file = f"{self.output_dir}/node_functions_{self.timestamp}.json"
        with open(node_functions_file, 'w') as f:
            json.dump(node_functions_log, f, indent=2)
        
        print(f"\nNode functions have been logged to: {node_functions_file}")
        if self.live_logging:
            print(f"Live execution logs will be written to: {self.jsonl_file}")
        print(f"State type: {self.state_type}")
        
        # Print a summary of what was logged
        print("\nLogged Node Functions:")
        print("---------------------")
        for node_name, data in node_functions_log["nodes"].items():
            if "error" not in data:
                print(f"- {node_name} ({data['metadata']['name']})")
                if data['metadata']['doc']:
                    print(f"  {data['metadata']['doc']}")
            else:
                print(f"- {node_name} (Error: {data['error']})")
    
    def compile(self) -> None:
        """Compile the workflow for execution."""
        self.chain = self.workflow.compile()
        print("✅ Workflow compiled with Spectra enhancements.")
        
        if self.live_logging:
            self._log_to_jsonl({
                "event": "workflow_compiled",
                "timestamp": datetime.now().isoformat()
            })
    
    def invoke(self, initial_state: Any) -> Any:
        """Invoke the workflow with the given initial state."""
        # self._clear_processed_data() # Removed from here
        
        if self.chain is None:
            self.compile() # Compile if not already compiled
        
        # Check again after attempting compilation
        if self.chain is None:
            raise RuntimeError("Workflow chain is not compiled and compilation failed.")

        # Reset current node for this run
        if hasattr(self, 'callback_handler') and self.callback_handler:
            self.callback_handler.current_node = None 

        print(f"🚀 Invoking workflow with initial state: {initial_state}")
        
        config = {}
        if self.capture_thinking and hasattr(self, 'callback_handler') and self.callback_handler:
            config['callbacks'] = [self.callback_handler]
            
        try:
            result = self.chain.invoke(initial_state, config=config)
            return result
        finally:
            self.flush_pending_logs() # Ensure all logs are sent before returning
    
    def run(self, initial_state: Any) -> Any:
        """Run the workflow with the given initial state (often an alias for invoke)."""
        # This method will now directly call invoke, 
        # which handles compiling and then invoking.
        # self._clear_processed_data() # Removed from here
        return self.invoke(initial_state)
    
    def set_current_node(self, node_name: str):
        """Set the current node context for thinking capture."""
        if self.capture_thinking and hasattr(self, 'callback_handler'):
            self.callback_handler.current_node = node_name
    
    def flush_pending_logs(self):
        """Immediately send any pending logs to the API if auto_process is enabled."""
        # Wait for any active timer to finish its attempt before force flushing
        # This is a simple way to avoid stepping on an active timer's attempt.
        # A more robust solution might involve joining the timer thread if it's alive.
        if self._api_call_timer and self._api_call_timer.is_alive():
            print("⏳ Waiting for scheduled API call to complete before flushing...")
            self._api_call_timer.join(timeout=self.api_cooldown + 1.0) # Wait for a bit longer than cooldown

        with self.log_buffer_lock:  # Protect access to pending_logs
            has_pending = bool(self.pending_logs)
        
        if has_pending and self.auto_process:
            print(f"🔄 Flushing {len(self.pending_logs)} remaining logs to API...")
            self._call_process_logs_api()
            # Give the API thread a moment to send, especially since it's non-daemon now
            # This is a bit of a heuristic. A more deterministic way would be to join the api_thread
            # if we had a direct reference to it and knew it was the *last* one.
            time.sleep(0.5) # Small delay to allow network I/O
        elif not self.auto_process and has_pending:
            print(f"ℹ️ Auto-processing disabled. {len(self.pending_logs)} logs remain in buffer and were not flushed to API.")
    
    def __del__(self):
        # Attempt to flush logs when the Spectra instance is garbage collected
        # This is a best-effort attempt, especially for scripts ending abruptly.
        print("Spectra instance finalizing. Attempting to flush any remaining logs...")
        self.flush_pending_logs()
        # Ensure any non-daemon timer is cancelled if still alive
        if self._api_call_timer and self._api_call_timer.is_alive():
            self._api_call_timer.cancel()
            print("Cancelled active API call timer during finalization.")

    def disable_auto_processing(self):
        """Disable automatic API processing of logs."""
        self.auto_process = False
        print("⏸️ Auto-processing of logs disabled")
    
    def enable_auto_processing(self):
        """Enable automatic API processing of logs."""
        self.auto_process = True
        print("▶️ Auto-processing of logs enabled") 