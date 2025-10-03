from langgraph.graph import StateGraph, START, END
from .AgentState import AgentState
from tools.tool_definitions import SimpleQueryPlanner, ReasoningEngine
from langgraph.checkpoint.memory import MemorySaver


def create_workflow_nodes(bedrock_llm, simple_change_tool, simple_cmdb_tool, simple_incident_tool):
    """Create workflow nodes with dependencies injected"""
    
    def simple_planner_node(state: AgentState) -> AgentState:
        """Create execution plan with conversation history"""
        state["current_step"] = "planning"
        state["messages"].append("📋 Planning execution...")
        
        try:
            planner = SimpleQueryPlanner(bedrock_llm)
            execution_plan = planner.create_execution_plan(
                state["user_input"], 
                state.get("conversation_history", [])
            )
            state["execution_plan"] = execution_plan
            state["current_plan_index"] = 0
            state["messages"].append(f"Created plan with {len(execution_plan)} steps")
            
            print("-" * 60,f"\n Execution plan :")
            for step in execution_plan:
                print(f"Step {step['step']}: {step['tool']}")
                print(f"  Purpose: {step['purpose']}")
            
        except Exception as e:
            state["messages"].append(f"Planning error: {str(e)}")
            # Simple fallback
            state["execution_plan"] = [{
                "step": 1,
                "tool": "cmdb_configuration", 
                "purpose": "general search",
                "query": state["user_input"]
            }]
            state["current_plan_index"] = 0
        
        return state

    def simple_executor_node(state: AgentState) -> AgentState:
        """Execute current step with conversation history"""
        state["current_step"] = "executing"
        
        if state["current_plan_index"] >= len(state["execution_plan"]):
            state["current_step"] = "execution_complete"
            return state
        
        current_step = state["execution_plan"][state["current_plan_index"]]
        tool_name = current_step["tool"]
        
        state["messages"].append(f"🔍 Executing step {current_step['step']}: {tool_name}")
        
        try:
            # Select tool
            if tool_name == "change_management":
                tool = simple_change_tool
            elif tool_name == "cmdb_configuration":
                tool = simple_cmdb_tool
            elif tool_name == "incident_management":
                tool = simple_incident_tool
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Execute with context AND conversation history
            result = tool.execute(
                current_step["purpose"],
                state["context_data"],
                state.get("conversation_history", [])
            )
            
            # Store result and update context
            state["intermediate_results"].append(result)
            
            if result["success"]:
                # Add analysis to context for next steps
                state["context_data"][f"step_{current_step['step']}_analysis"] = result["analysis"]
                print("-"*60, f"\nAnalyis of step_{current_step['step']}= {tool_name} : \n {result["analysis"]}")

                state["messages"].append(f"✅ Step {current_step['step']} completed successfully")
            else:
                state["messages"].append(f"❌ Step {current_step['step']} failed: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            state["messages"].append(f"Execution error: {str(e)}")
            state["intermediate_results"].append({
                "tool_name": tool_name,
                "success": False,
                "error": str(e),
                "analysis": f"Failed to execute {tool_name}: {str(e)}",
                "confidence": 0.0
            })

        return state

    def simple_reasoning_node(state: AgentState) -> AgentState:
        """Simple reasoning logic"""
        state["current_step"] = "reasoning"
        state["total_iterations"] = state.get("total_iterations", 0) + 1
        
        # Safety check
        if state["total_iterations"] > state.get("max_total_iterations", 10):
            state["messages"].append("⚠️ Max iterations reached")
            state["current_plan_index"] = len(state["execution_plan"])
            return state
        
        # Simple logic: if last step failed and we have alternatives, try alternative
        if state["intermediate_results"]:
            last_result = state["intermediate_results"][-1]
            current_step_index = state["current_plan_index"]
            
            step_attempts = state["step_attempt_counts"].get(current_step_index, 0)
            
            if not last_result["success"] and step_attempts < 2:  # Max 2 attempts per step
                # Try next tool as alternative
                tools = ["change_management", "cmdb_configuration", "incident_management"]
                current_tool = state["execution_plan"][current_step_index]["tool"]
                
                if current_tool in tools:
                    current_idx = tools.index(current_tool)
                    next_idx = (current_idx + 1) % len(tools)
                    state["execution_plan"][current_step_index]["tool"] = tools[next_idx]
                    state["step_attempt_counts"][current_step_index] = step_attempts + 1
                    state["messages"].append(f"🔄 Trying alternative tool: {tools[next_idx]}")
                    return state
        
        # Move to next step
        state["current_plan_index"] += 1
        return state

    def simple_synthesis_node(state: AgentState) -> AgentState:
        """Synthesize final response with conversation history"""
        state["current_step"] = "synthesizing"
        
        # Collect all analyses
        analyses = []
        for result in state["intermediate_results"]:
            if result.get("success") and result.get("analysis"):
                analyses.append(f"From {result['tool_name']}: {result['analysis']}")
        
        if not analyses:
            state["final_response"] = "I couldn't find relevant information to answer your question. Please try rephrasing or asking about specific change IDs, application names, or incident numbers."
            return state
        
        # Format conversation history for synthesis
        history_str = ""
        if state.get("conversation_history"):
            history_str = "\nPrevious conversation context:"
            for msg in state["conversation_history"][-4:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")[:200]
                history_str += f"\n{role.title()}: {content}"
        
        synthesis_prompt = f"""
        Based on the information gathered, provide a comprehensive answer to the user's question.
        
        {history_str}
        
        User's current question: "{state['user_input']}"
        
        Information found:
        {chr(10).join(analyses)}
        
        IMPORTANT: If the user's question refers to something from previous conversation (using "it", "its", "that", etc.), use the conversation history to understand the reference and provide a complete answer.
        
        Response:
        """
        
        try:
            final_response = bedrock_llm.invoke(synthesis_prompt).content
            state["final_response"] = final_response
            state["messages"].append("✅ Response synthesized")
        except Exception as e:
            state["final_response"] = f"I found some information but had trouble synthesizing the final response: {str(e)}"
            state["messages"].append(f"Synthesis error: {str(e)}")
        
        return state
    
    return simple_planner_node, simple_executor_node, simple_reasoning_node, simple_synthesis_node


def should_continue_simple(state: AgentState) -> str:
    """Simple routing logic"""
    if state.get("total_iterations", 0) > 10:
        return "synthesize"
    if state.get("current_plan_index", 0) < len(state.get("execution_plan", [])):
        return "execute"
    return "synthesize"


def create_workflow(memory, bedrock_llm, simple_change_tool, simple_cmdb_tool, simple_incident_tool):
    """Create and compile the workflow"""
    
    # Get workflow nodes
    simple_planner_node, simple_executor_node, simple_reasoning_node, simple_synthesis_node = create_workflow_nodes(
        bedrock_llm, simple_change_tool, simple_cmdb_tool, simple_incident_tool
    )
    
    # Create workflow
    simple_workflow = StateGraph(AgentState)

    # Add nodes (no replanner)
    simple_workflow.add_node("planner", simple_planner_node)
    simple_workflow.add_node("executor", simple_executor_node)
    simple_workflow.add_node("reasoner", simple_reasoning_node)
    simple_workflow.add_node("synthesizer", simple_synthesis_node)

    # Add edges
    simple_workflow.add_edge(START, "planner")
    simple_workflow.add_edge("planner", "executor")
    simple_workflow.add_edge("executor", "reasoner")

    # Conditional routing
    simple_workflow.add_conditional_edges(
        "reasoner",
        should_continue_simple,
        {
            "execute": "executor",
            "synthesize": "synthesizer"
        }
    )

    simple_workflow.add_edge("synthesizer", END)

    # Compile
    memory = MemorySaver()
    simple_app = simple_workflow.compile(checkpointer=memory)

    
    print("✅ Simplified Enhanced Multi-Step Reasoning Agent created!")
    
    return simple_app