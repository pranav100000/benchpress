# UI/Display constants
PANEL_WIDTH = None
QUESTION_BORDER_PANEL_COLOR = "blue"
CORRECT_ANSWER_BORDER_PANEL_COLOR = "green"
INCORRECT_ANSWER_BORDER_PANEL_COLOR = "red"
MODEL_RESPONSE_BORDER_PANEL_COLOR = "yellow"
DEBUG_BORDER_PANEL_COLOR = "red"

# Other configuration constants
MAX_REFRESH_RATE = 10

# Timeout constants (in seconds)
DEFAULT_TIMEOUT_SECONDS = 30
API_REQUEST_TIMEOUT = 60.0  # Default timeout for API requests
STREAMING_TIMEOUT = 120.0   # Longer timeout for streaming responses
TASK_TIMEOUT = 180.0        # Maximum time to wait for a task to complete
PARALLEL_WAIT_TIMEOUT = 60.0  # Timeout for asyncio.wait in parallel mode
