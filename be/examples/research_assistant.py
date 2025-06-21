"""
Advanced Research Assistant Workflow

This demonstrates a sophisticated LangGraph workflow with:
- Multiple specialized agents with different roles
- Complex state management with validation
- Conditional routing based on quality checks
- Tool integration (web search simulation)
- Error handling and retry logic
- Rich LLM interactions perfect for thinking capture

Run with: python examples/research_assistant.py
"""

from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.tools import tool
from spectra import Spectra
from dotenv import load_dotenv
import json
import random
import time

load_dotenv()

# Complex State Definition
class ResearchState(TypedDict):
    # Input
    query: Annotated[str, "The research question to investigate"]
    
    # Query Analysis
    research_plan: Annotated[Dict[str, Any], "Structured research plan"]
    key_topics: Annotated[List[str], "Key topics to research"]
    complexity_level: Annotated[str, "Simple/Medium/Complex"]
    
    # Search Results
    search_results: Annotated[List[Dict], "Raw search results"]
    processed_sources: Annotated[List[Dict], "Analyzed and processed sources"]
    
    # Analysis
    key_findings: Annotated[List[str], "Important findings from research"]
    evidence_quality: Annotated[str, "Assessment of evidence quality"]
    conflicting_info: Annotated[List[str], "Any conflicting information found"]
    
    # Synthesis
    insights: Annotated[List[str], "Key insights and connections"]
    recommendations: Annotated[List[str], "Actionable recommendations"]
    
    # Quality Control
    fact_check_results: Annotated[List[Dict], "Fact-checking results"]
    confidence_score: Annotated[float, "Overall confidence in findings (0-1)"]
    
    # Final Output
    final_report: Annotated[str, "Comprehensive research report"]
    executive_summary: Annotated[str, "Brief executive summary"]
    
    # Workflow Control
    next_step: Annotated[str, "Next workflow step"]
    retry_count: Annotated[int, "Number of retries for current step"]
    errors: Annotated[List[str], "Any errors encountered"]

# Initialize different LLMs for different tasks
analyzer_llm = ChatOpenAI(model="gpt-4", temperature=0.1)  # Precise analysis
researcher_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)  # Balanced research
synthesizer_llm = ChatOpenAI(model="gpt-4", temperature=0.7)  # Creative synthesis
fact_checker_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)  # Strict fact-checking

# Simulated Tools
@tool
def web_search(query: str, num_results: int = 5) -> List[Dict]:
    """Simulated web search tool. In production, use real search APIs."""
    # Simulate realistic search results with delays
    time.sleep(1)  # Simulate network delay
    
    fake_results = [
        {
            "title": f"Research on {query}: Academic Perspective",
            "url": "https://academic-source.edu/research",
            "snippet": f"Recent studies on {query} show significant developments in the field. This comprehensive analysis covers multiple aspects and provides evidence-based insights.",
            "source_type": "academic",
            "credibility": 0.9
        },
        {
            "title": f"{query}: Industry Report 2024",
            "url": "https://industry-report.com/analysis",
            "snippet": f"Industry analysis of {query} reveals current trends and future projections. Market data suggests continued growth and innovation.",
            "source_type": "industry",
            "credibility": 0.8
        },
        {
            "title": f"Expert Opinion: Understanding {query}",
            "url": "https://expert-blog.com/opinion",
            "snippet": f"Leading experts discuss the implications of {query} and share their perspectives on best practices and future directions.",
            "source_type": "expert_opinion",
            "credibility": 0.7
        },
        {
            "title": f"Case Study: {query} Implementation",
            "url": "https://case-study.org/implementation",
            "snippet": f"Real-world case study examining the practical implementation of {query} strategies and their outcomes.",
            "source_type": "case_study",
            "credibility": 0.8
        },
        {
            "title": f"News: Latest Developments in {query}",
            "url": "https://news-source.com/latest",
            "snippet": f"Breaking news and recent developments related to {query}. Current events and their potential impact on the field.",
            "source_type": "news",
            "credibility": 0.6
        }
    ]
    
    return fake_results[:num_results]

# Agent Definitions
def query_analyzer(state: ResearchState) -> ResearchState:
    """Analyzes the research query and creates a structured research plan."""
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research planning expert. Your job is to analyze research questions and create comprehensive research plans.
        
        For each query, provide:
        1. A structured research plan with clear objectives
        2. Key topics that need investigation
        3. Complexity assessment (Simple/Medium/Complex)
        4. Suggested research approaches
        
        Be thorough and consider multiple angles of investigation."""),
        ("user", """Research Query: {query}
        
        Please analyze this query and create a detailed research plan. Return your response in this JSON format:
        {{
            "research_plan": {{
                "main_objective": "...",
                "sub_objectives": ["...", "..."],
                "research_approaches": ["...", "..."],
                "success_criteria": ["...", "..."]
            }},
            "key_topics": ["topic1", "topic2", "topic3"],
            "complexity_level": "Simple/Medium/Complex",
            "estimated_time": "X hours",
            "reasoning": "Why this approach is optimal..."
        }}""")
    ])
    
    try:
        chain = analysis_prompt | analyzer_llm
        response = chain.invoke({"query": state["query"]})
        
        # Parse the JSON response
        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            analysis = {
                "research_plan": {
                    "main_objective": f"Research and analyze: {state['query']}",
                    "sub_objectives": ["Gather relevant information", "Analyze findings", "Synthesize insights"],
                    "research_approaches": ["Literature review", "Expert analysis", "Data synthesis"],
                    "success_criteria": ["Comprehensive coverage", "Evidence-based conclusions"]
                },
                "key_topics": [topic.strip() for topic in state["query"].split()[:3]],
                "complexity_level": "Medium",
                "estimated_time": "2-3 hours",
                "reasoning": "Standard research approach for comprehensive analysis"
            }
        
        return {
            **state,
            "research_plan": analysis["research_plan"],
            "key_topics": analysis["key_topics"],
            "complexity_level": analysis["complexity_level"],
            "next_step": "search"
        }
        
    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Query analysis error: {str(e)}"],
            "next_step": "error"
        }

def web_researcher(state: ResearchState) -> ResearchState:
    """Conducts web research using search tools and processes results."""
    
    search_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research specialist. Given a set of search results, extract and organize the most relevant information.
        
        Your tasks:
        1. Assess the credibility of each source
        2. Extract key information relevant to the research objectives
        3. Identify any conflicting information
        4. Organize findings by topic/theme
        
        Be critical and thorough in your analysis."""),
        ("user", """Research Query: {query}
        Key Topics: {key_topics}
        
        Search Results:
        {search_results}
        
        Please process these results and extract relevant information. Focus on factual content and credible sources.""")
    ])
    
    try:
        # Conduct searches for each key topic
        all_results = []
        for topic in state["key_topics"]:
            search_query = f"{state['query']} {topic}"
            results = web_search(search_query, num_results=3)
            all_results.extend(results)
        
        # Process results with LLM
        chain = search_prompt | researcher_llm
        response = chain.invoke({
            "query": state["query"],
            "key_topics": ", ".join(state["key_topics"]),
            "search_results": json.dumps(all_results, indent=2)
        })
        
        # Create processed sources
        processed_sources = []
        for result in all_results:
            processed_sources.append({
                "title": result["title"],
                "url": result["url"],
                "content": result["snippet"],
                "credibility": result["credibility"],
                "source_type": result["source_type"],
                "relevance_score": random.uniform(0.6, 0.95)  # Simulated relevance
            })
        
        return {
            **state,
            "search_results": all_results,
            "processed_sources": processed_sources,
            "next_step": "analyze"
        }
        
    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Research error: {str(e)}"],
            "next_step": "error"
        }

def content_analyzer(state: ResearchState) -> ResearchState:
    """Analyzes processed sources and extracts key findings."""
    
    analysis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a content analysis expert. Your job is to analyze research sources and extract key findings.
        
        For each analysis:
        1. Identify the most important findings from all sources
        2. Assess the overall quality of evidence
        3. Note any conflicting information
        4. Highlight gaps in the research
        
        Be objective and evidence-based in your analysis."""),
        ("user", """Research Query: {query}
        
        Processed Sources:
        {sources}
        
        Please analyze these sources and provide:
        1. Key findings (list of important discoveries/facts)
        2. Evidence quality assessment (High/Medium/Low)
        3. Any conflicting information found
        4. Research gaps or limitations identified""")
    ])
    
    try:
        chain = analysis_prompt | analyzer_llm
        response = chain.invoke({
            "query": state["query"],
            "sources": json.dumps(state["processed_sources"], indent=2)
        })
        
        # Extract key findings from response
        content = response.content
        lines = content.split('\n')
        
        key_findings = []
        conflicting_info = []
        evidence_quality = "Medium"  # Default
        
        current_section = None
        for line in lines:
            line = line.strip()
            if "key findings" in line.lower():
                current_section = "findings"
            elif "conflicting" in line.lower():
                current_section = "conflicts"
            elif "evidence quality" in line.lower():
                if "high" in line.lower():
                    evidence_quality = "High"
                elif "low" in line.lower():
                    evidence_quality = "Low"
            elif line.startswith(('-', '•', '1.', '2.', '3.')):
                if current_section == "findings":
                    key_findings.append(line)
                elif current_section == "conflicts":
                    conflicting_info.append(line)
        
        return {
            **state,
            "key_findings": key_findings or ["Analysis of research sources completed"],
            "evidence_quality": evidence_quality,
            "conflicting_info": conflicting_info,
            "next_step": "synthesize"
        }
        
    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Analysis error: {str(e)}"],
            "next_step": "error"
        }

def synthesis_agent(state: ResearchState) -> ResearchState:
    """Synthesizes findings into insights and recommendations."""
    
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a strategic synthesis expert. Your role is to take research findings and create actionable insights.
        
        Your tasks:
        1. Connect findings to identify patterns and trends
        2. Generate strategic insights that go beyond individual findings
        3. Create actionable recommendations based on the evidence
        4. Consider practical implications and next steps
        
        Be creative in making connections while staying grounded in evidence."""),
        ("user", """Research Query: {query}
        
        Key Findings:
        {findings}
        
        Evidence Quality: {evidence_quality}
        
        Conflicting Information:
        {conflicts}
        
        Please synthesize this information into:
        1. Strategic insights (connections and patterns you see)
        2. Actionable recommendations (what should be done based on this research)
        3. Confidence assessment for your synthesis""")
    ])
    
    try:
        chain = synthesis_prompt | synthesizer_llm
        response = chain.invoke({
            "query": state["query"],
            "findings": "\n".join(state["key_findings"]),
            "evidence_quality": state["evidence_quality"],
            "conflicts": "\n".join(state["conflicting_info"]) if state["conflicting_info"] else "None identified"
        })
        
        content = response.content
        lines = content.split('\n')
        
        insights = []
        recommendations = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "insights" in line.lower():
                current_section = "insights"
            elif "recommendations" in line.lower():
                current_section = "recommendations"
            elif line.startswith(('-', '•', '1.', '2.', '3.')):
                if current_section == "insights":
                    insights.append(line)
                elif current_section == "recommendations":
                    recommendations.append(line)
        
        return {
            **state,
            "insights": insights or ["Synthesis of research findings completed"],
            "recommendations": recommendations or ["Further research recommended"],
            "next_step": "fact_check"
        }
        
    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Synthesis error: {str(e)}"],
            "next_step": "error"
        }

def fact_checker(state: ResearchState) -> ResearchState:
    """Performs fact-checking and quality assessment."""
    
    fact_check_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a rigorous fact-checker. Your job is to assess the reliability and accuracy of research findings.
        
        For each fact-check:
        1. Evaluate the credibility of sources
        2. Check for logical consistency
        3. Identify potential biases or limitations
        4. Assign confidence scores to key claims
        
        Be thorough and conservative in your assessments."""),
        ("user", """Research Query: {query}
        
        Key Findings:
        {findings}
        
        Insights:
        {insights}
        
        Source Quality: {evidence_quality}
        
        Please fact-check this research and provide:
        1. Overall confidence score (0.0 to 1.0)
        2. Any reliability concerns
        3. Suggestions for improving confidence""")
    ])
    
    try:
        chain = fact_check_prompt | fact_checker_llm
        response = chain.invoke({
            "query": state["query"],
            "findings": "\n".join(state["key_findings"]),
            "insights": "\n".join(state["insights"]),
            "evidence_quality": state["evidence_quality"]
        })
        
        # Extract confidence score and create fact-check results
        content = response.content.lower()
        confidence_score = 0.7  # Default
        
        # Try to extract confidence score
        for line in content.split('\n'):
            if 'confidence' in line and any(char.isdigit() for char in line):
                import re
                numbers = re.findall(r'0\.\d+|\d+\.\d+', line)
                if numbers:
                    try:
                        score = float(numbers[0])
                        if 0 <= score <= 1:
                            confidence_score = score
                        elif score > 1:
                            confidence_score = score / 100  # Convert percentage
                    except ValueError:
                        pass
        
        fact_check_results = [
            {
                "aspect": "Source Credibility",
                "assessment": "Generally reliable sources with academic and industry backing",
                "confidence": confidence_score
            },
            {
                "aspect": "Logical Consistency",
                "assessment": "Findings are logically consistent with established knowledge",
                "confidence": confidence_score
            }
        ]
        
        return {
            **state,
            "fact_check_results": fact_check_results,
            "confidence_score": confidence_score,
            "next_step": "generate_report"
        }
        
    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Fact-check error: {str(e)}"],
            "confidence_score": 0.5,
            "next_step": "generate_report"
        }

def report_generator(state: ResearchState) -> ResearchState:
    """Generates the final comprehensive research report."""
    
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research report writer. Create comprehensive, well-structured reports that are both thorough and accessible.
        
        Your report should include:
        1. Executive Summary (brief overview)
        2. Research Methodology
        3. Key Findings
        4. Analysis and Insights
        5. Recommendations
        6. Confidence Assessment
        7. Limitations and Future Research
        
        Use clear, professional language and logical organization."""),
        ("user", """Research Query: {query}
        
        Research Plan: {research_plan}
        Key Findings: {findings}
        Insights: {insights}
        Recommendations: {recommendations}
        Confidence Score: {confidence}
        
        Please generate a comprehensive research report covering all aspects of this investigation.""")
    ])
    
    try:
        chain = report_prompt | synthesizer_llm
        response = chain.invoke({
            "query": state["query"],
            "research_plan": json.dumps(state["research_plan"], indent=2),
            "findings": "\n".join(state["key_findings"]),
            "insights": "\n".join(state["insights"]),
            "recommendations": "\n".join(state["recommendations"]),
            "confidence": f"{state['confidence_score']:.2f}"
        })
        
        final_report = response.content
        
        # Generate executive summary
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Create a brief executive summary (2-3 paragraphs) of this research report."),
            ("user", "Full Report:\n{report}")
        ])
        
        summary_chain = summary_prompt | analyzer_llm
        summary_response = summary_chain.invoke({"report": final_report})
        executive_summary = summary_response.content
        
        return {
            **state,
            "final_report": final_report,
            "executive_summary": executive_summary,
            "next_step": "complete"
        }
        
    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [f"Report generation error: {str(e)}"],
            "final_report": "Error generating report. Please see error logs.",
            "executive_summary": "Report generation failed due to technical issues.",
            "next_step": "complete"
        }

# Routing Functions
def should_continue(state: ResearchState) -> str:
    """Determines the next step in the workflow."""
    next_step = state.get("next_step", "analyze")
    
    # Handle errors with retry logic
    if next_step == "error":
        retry_count = state.get("retry_count", 0)
        if retry_count < 2:  # Allow 2 retries
            return "analyze"  # Retry from analysis
        else:
            return "generate_report"  # Skip to report with partial results
    
    # Normal flow
    step_mapping = {
        "search": "search",
        "analyze": "analyze", 
        "synthesize": "synthesize",
        "fact_check": "fact_check",
        "generate_report": "generate_report",
        "complete": END
    }
    
    return step_mapping.get(next_step, END)

def create_research_workflow():
    """Creates the advanced research assistant workflow."""
    workflow = StateGraph(ResearchState)
    
    # Add all agent nodes
    workflow.add_node("query_analysis", query_analyzer)
    workflow.add_node("search", web_researcher)
    workflow.add_node("analyze", content_analyzer)
    workflow.add_node("synthesize", synthesis_agent)
    workflow.add_node("fact_check", fact_checker)
    workflow.add_node("generate_report", report_generator)
    
    # Define the flow
    workflow.add_edge(START, "query_analysis")
    
    # Conditional routing based on state
    workflow.add_conditional_edges(
        "query_analysis",
        should_continue,
        {
            "search": "search",
            "error": "generate_report"
        }
    )
    
    workflow.add_conditional_edges(
        "search",
        should_continue,
        {
            "analyze": "analyze",
            "error": "generate_report"
        }
    )
    
    workflow.add_conditional_edges(
        "analyze",
        should_continue,
        {
            "synthesize": "synthesize",
            "error": "generate_report"
        }
    )
    
    workflow.add_conditional_edges(
        "synthesize",
        should_continue,
        {
            "fact_check": "fact_check",
            "error": "generate_report"
        }
    )
    
    workflow.add_conditional_edges(
        "fact_check",
        should_continue,
        {
            "generate_report": "generate_report"
        }
    )
    
    workflow.add_conditional_edges(
        "generate_report",
        should_continue,
        {
            END: END
        }
    )
    
    return workflow

if __name__ == "__main__":
    print("🔬 Advanced Research Assistant")
    print("=" * 50)
    print("Multi-agent research workflow with LLM thinking capture")
    print("This demonstrates complex state management, tool usage, and error handling")
    print()
    
    # Create the workflow
    workflow = create_research_workflow()
    
    # Initialize Spectra with full thinking capture
    spectra_logger = Spectra(workflow, capture_thinking=True)
    
    # Complex research queries that will showcase the system
    research_queries = [
        "What are the current trends and future prospects for artificial intelligence in healthcare, including benefits, challenges, and ethical considerations?",
        "How is renewable energy adoption affecting global energy markets, and what are the implications for traditional energy companies?",
        "What are the key factors driving remote work adoption in large corporations, and how is this trend impacting urban real estate markets?"
    ]
    
    for i, query in enumerate(research_queries, 1):
        print(f"\n📋 Research Query {i}:")
        print(f"'{query}'")
        print("\n🔄 Starting research workflow...")
        print("This may take a few minutes as we conduct thorough analysis...")
        
        # Initialize state
        initial_state = ResearchState(
            query=query,
            research_plan={},
            key_topics=[],
            complexity_level="Medium",
            search_results=[],
            processed_sources=[],
            key_findings=[],
            evidence_quality="Medium",
            conflicting_info=[],
            insights=[],
            recommendations=[],
            fact_check_results=[],
            confidence_score=0.0,
            final_report="",
            executive_summary="",
            next_step="search",
            retry_count=0,
            errors=[]
        )
        
        try:
            # Run the research workflow
            final_state = spectra_logger.run(initial_state)
            
            # Display results
            print(f"\n📊 Research Complete!")
            print(f"📈 Confidence Score: {final_state['confidence_score']:.1%}")
            print(f"🎯 Key Topics: {', '.join(final_state['key_topics'])}")
            print(f"📝 Sources Analyzed: {len(final_state['processed_sources'])}")
            
            print(f"\n📋 Executive Summary:")
            print("-" * 30)
            print(final_state['executive_summary'])
            
            print(f"\n💡 Key Insights:")
            for insight in final_state['insights'][:3]:  # Show top 3
                print(f"  • {insight}")
            
            print(f"\n🎯 Top Recommendations:")
            for rec in final_state['recommendations'][:3]:  # Show top 3
                print(f"  • {rec}")
            
            if final_state['errors']:
                print(f"\n⚠️  Errors encountered: {len(final_state['errors'])}")
            
        except Exception as e:
            print(f"\n❌ Research failed: {str(e)}")
        
        print("\n" + "="*80)
        
        # Only run one query for demo purposes
        if i == 1:
            break
    
    print(f"\n💾 Complete research logs available in:")
    print(f"   📄 JSON: Workflow structure and agent functions")
    print(f"   📊 JSONL: Live execution with detailed LLM thinking")
    print(f"   🧠 Thinking capture includes: prompts, responses, token usage, tool calls")
    print(f"\nThis demonstrates advanced LangGraph patterns:")
    print(f"  • Multi-agent collaboration")
    print(f"  • Complex state management") 
    print(f"  • Conditional routing & error handling")
    print(f"  • Tool integration")
    print(f"  • Rich LLM interactions") 