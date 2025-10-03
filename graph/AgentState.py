from typing import TypedDict, List, Dict, Any, Optional
 
class AgentState(TypedDict):
    messages: List[str]
    user_input: str
    current_step: str
    execution_plan: List[Dict[str, Any]]
    current_plan_index: int
    context_data: Dict[str, Any]
    intermediate_results: List[Dict[str, Any]]
    final_response: str
    error_recovery_attempts: int
    max_recovery_attempts: int
    thread_id: Optional[str]
    # Safety parameters
    step_attempt_counts: Dict[int, int]
    max_step_attempts: int
    total_iterations: int
    max_total_iterations: int
    # NEW: Conversation history
    conversation_history: List[Dict[str, str]]
    retrieved_raw_data: str 