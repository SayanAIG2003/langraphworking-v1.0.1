from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

class SmartToolWrapper:
    """Enhanced tool wrapper with conversation history support"""
    
    def __init__(self, retriever, llm, tool_name: str, data_description: str):
        self.retriever = retriever
        self.llm = llm
        self.tool_name = tool_name
        self.data_description = data_description
    
    def format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for LLM context"""
        if not conversation_history:
            return "\nNo previous conversation."
        
        history_str = "\nPrevious conversation context:"
        for msg in conversation_history[-6:]:  # Last 6 messages to keep context manageable
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Truncate long messages
            history_str += f"\n{role.title()}: {content}"
        
        return history_str
    
    def generate_retriever_query(self, user_query: str, context: Dict[str, Any], history_str: str) -> str:
        """Generate query with conversation history context"""
        
        context_str = ""
        if context:
            context_str = f"\nContext from previous steps: {str(context)[:500]}"
        
        prompt = f"""
        Generate a single line search question for self query retriver on {self.tool_name} database.
        
        Note: The query should be human like in english.
        
        {history_str}
        
        You need to generate a specific/concise question out of user query for the {self.tool_name} self query retriever.
        
        {self.data_description}
        
        User's current question: "{user_query}"
        {context_str}
        
        IMPORTANT: If the user uses pronouns like "it", "its", "that", "this", "them", etc., use the conversation history to understand what they're referring to. Replace pronouns with specific names, IDs, or entities from previous context.
        
        Based on the user's question, conversation history, and any context, create a focused human-like question for the self query retriever that will help find the most relevant information from this data source.
        
        Be specific but not overly narrow. Focus on key identifiers, names, IDs, or criteria. The query should be 1 line.

        Use GEAR_ID from context if available instead of application names
        
        Search question:
        """
        
        response = self.llm.invoke(prompt).content.strip()

        # Extract only the first meaningful line, remove any prefix
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            query = lines[0]
            # Remove common prefixes if they exist
            prefixes_to_remove = ['Query:', 'Search:', 'Question:', 'Find:', 'Get:', 'Retrieve:']
            for prefix in prefixes_to_remove:
                if query.startswith(prefix):
                    query = query[len(prefix):].strip()
            return query
        
        return response.strip()
    
    def analyze_and_extract(self, retrieved_docs: str, user_query: str, context: Dict[str, Any], history_str: str) -> str:
        """Analyze documents with conversation history context"""
        
        context_str = ""
        if context:
            context_str = f"\nContext: {str(context)[:500]}"
        
        prompt = f"""
        Analyze the following data and extract information relevant to the user's question.
        
        {history_str}
        
        User's current question: "{user_query}"
        {context_str}
        
        Retrieved data:
        {str(retrieved_docs)[:2000]}
        
        IMPORTANT: If the user's question refers to something from previous conversation (using "it", "its", "that", etc.), use the conversation history to understand the reference.
        
        Provide your analysis in this exact format:

        **Answer:** [answer to user's question]
        
        **GEAR_IDs Found:** [list all unique GEAR_IDs found - this is MANDATORY]
        
        **Key Identifiers:** [other important IDs, names, numbers]
        
        **Summary:** [concise summary of findings]
        
        CRITICAL: Always extract ALL GEAR_IDs from the data or context if provided. Other tools will use these GEAR_IDs to find related information across databases.
        
        Analysis:
        """
        
        response = self.llm.invoke(prompt).content.strip()
        return response
    
    def execute(self, user_query: str, context: Dict[str, Any], conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Execute with conversation history support"""
        try:
            history_str = self.format_conversation_history(conversation_history or [])

            # Step 1: Generate retriever query using LLM with history
            retriever_query = self.generate_retriever_query(user_query, context, history_str)
            print("-"*80,f"\n{self.tool_name} query: {retriever_query}")

            self.retriever.query_filter_construct(retriever_query)
            
            # Step 2: Execute retriever
            retrieved_docs = self.retriever.retrieve_docs(retriever_query)
            
            # Step 3: Analyze and extract using LLM with history
            analysis = self.analyze_and_extract(retrieved_docs, user_query, context, history_str)
            
            # Simple success evaluation
            success = len(analysis.strip()) > 50  # Basic check
            confidence = 0.8 if "found" in analysis.lower() or "identified" in analysis.lower() else 0.5
            
            return {
                "tool_name": self.tool_name,
                "success": success,
                "retriever_query": retriever_query,
                "raw_docs": str(retrieved_docs)[:1000],  # Truncated for storage
                "analysis": analysis,
                "confidence": confidence,
                "error": None
            }
            
        except Exception as e:
            return {
                "tool_name": self.tool_name,
                "success": False,
                "error": str(e),
                "analysis": f"Error retrieving {self.tool_name} data: {str(e)}",
                "confidence": 0.0
            }


class SimpleQueryPlanner:
    """Enhanced planner with conversation history support"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> str:
        """Format conversation history for planning context"""
        if not conversation_history:
            return "\nNo previous conversation."
        
        history_str = "\nPrevious conversation:"
        for msg in conversation_history[-4:]:  # Last 4 messages for planning
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:300]
            history_str += f"\n{role.title()}: {content}"
        
        return history_str
    
    def create_execution_plan(self, user_query: str, conversation_history: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Create execution plan with conversation history context"""
        
        history_str = self.format_conversation_history(conversation_history or [])
        
        planning_prompt = f"""
        Analyze this IT service management question and decide which data sources to query and in what order:
        
        {history_str}

       
        Current question: "{user_query}"
       
        Available data sources:
        1. change_management - Change requests, deployments, maintenance. The fields that this data has :Change ID,Priority,Opened Date(day, month, year separately),Completed Date(day, month, year separately),Status,Infra Vs App,Configuration Item,Assignment Group,GEAR_ID,APPLICATION_NAME,Segment,Region,Close Code,Trigger,Description,Change Type

        2. cmdb_configuration - Servers, Configuration item, applications, gear IDs, support groups, application and configuration item mappings. The fields this data has : SERVER_NAME,APPLICATION_NAME,GEAR_ID,CI_CATEGORY,DV_INSTALL_STATUS,IP_ADDRESS,DV_OPERATIONAL_STATUS,DV_SUPPORT_GROUP
        
        3. incident_management - Incidents ticket details like : Number	Priority	State	GEAR_ID	APPLICATION_NAME	CI	Assignment Group	Assigned to	Knowledge Article Used	Short description	Tags	Description 	Closed(day, month, year separately)		Opened(day, month, year separately)	Closed by(person name)	Impact	Urgency
       
        IMPORTANT: If the user's current question refers to something from previous conversation (using "it", "its", "that", "this", etc.), use the conversation history to understand what they're referring to.
        
        Consider:
        - What information is needed to answer the question?
        - What order makes sense? (e.g., get change details first, then find related servers)
        - Which sources are most likely to have the answer?
       
        Create a simple step-by-step plan. Usually 1-3 steps is enough. ! tool should be used once only

        What to find to should be very clear and concise for each tool. And "what to find" of one tool should not coincide with other tool "what to find".
       
        Respond with:
        Step 1: [tool_name] - [what to find]
        Step 2: [tool_name] - [what to find]
        Step 3: [tool_name] - [what to find]
       
        Plan:
        """
        
        response = self.llm.invoke(planning_prompt).content
        
        # Parse the simple response into plan structure
        plan = []
        lines = response.split('\n')
        
        print("Smart planner llm reponse : \n ")
        for i, line in enumerate(lines):
            print(line)
            if line.strip().startswith('Step'):
                parts = line.split(':')
                if len(parts) >= 2:
                    content = parts[1].strip()
                    if '-' in content:
                        tool_part, purpose = content.split('-', 1)
                        tool_name = tool_part.strip()
                        
                        # Map tool names
                        if 'change' in tool_name.lower():
                            tool_name = 'change_management'
                        elif 'cmdb' in tool_name.lower() or 'config' in tool_name.lower():
                            tool_name = 'cmdb_configuration' 
                        elif 'incident' in tool_name.lower():
                            tool_name = 'incident_management'
                        
                        plan.append({
                            "step": i + 1,
                            "tool": tool_name,
                            "purpose": purpose.strip(),
                            "query": user_query  # Use original query, tool will adapt it
                        })
        
        # Fallback if parsing fails
        if not plan:
            plan = [{
                "step": 1,
                "tool": "cmdb_configuration",
                "purpose": "general lookup",
                "query": user_query
            }]
        
        return plan
    
class ReasoningEngine:
    """Handles reasoning between tool executions"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def analyze_step_result(self, step_result: Dict[str, Any], current_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze result from a tool execution"""
        
        # Simplified analysis to avoid JSON parsing issues
        success = step_result.get("success", False)
        confidence = step_result.get("output", {}).get("confidence", 0.5)
        
        if not success:
            return {
                "success": False,
                "key_info_obtained": [],
                "missing_info": [f"Tool {step_result.get('tool_name', 'unknown')} failed"],
                "next_action": "try_alternative",
                "next_queries": {},
                "confidence": 0.1,
                "reasoning": f"Tool execution failed: {step_result.get('error', 'Unknown error')}"
            }
        
        if confidence < 0.3:
            return {
                "success": True,
                "key_info_obtained": ["Low confidence result"],
                "missing_info": ["Need more reliable information"],
                "next_action": "try_alternative",
                "next_queries": {},
                "confidence": confidence,
                "reasoning": "Low confidence result, trying alternative approach"
            }
        
        # Success case
        return {
            "success": True,
            "key_info_obtained": ["Tool executed successfully"],
            "missing_info": [],
            "next_action": "continue",
            "next_queries": {},
            "confidence": confidence,
            "reasoning": "Tool executed successfully, continuing to next step"
        }