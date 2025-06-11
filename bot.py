from flask import Flask, request, jsonify
import requests
import json
import logging
from flask_cors import CORS


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# The Tool
def chromaQuery(q):
    logger.info(f"Tool 'query' called with: {q}")
    #call the chroma shit. 
    return q #return the chroma shit-


# Create a mapping between tool names and functions
tool_map = {
    "chromaQuery": chromaQuery,
}

# Define tools in JSON format (OpenAI function calling format)
tools = [
    {
        "type": "function",
        "function": {
            "name": "chromaQuery",
            "description": "Searches information in a database containing specific explanations about LP specifics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User query"}
                },
                "required": ["query"]
            }
        }
    }
]

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({'error': 'Invalid input format. Expected JSON with "input" field.'}), 400

    try:
        user_query = data['input']
        logger.info(f"Received query: {user_query}")
        
        # Initialize conversation
        messages = [
            {"role": "system", "content": "You are a friendly assistant with expert knowledge in Tietoevry's Loan Process solution."},
            {"role": "user", "content": user_query}
        ]
        
        # Maximum number of iterations to prevent infinite loops
        max_iterations = 5
        iterations = 0
        
        # Start conversation loop
        while iterations < max_iterations:
            iterations += 1
            
            # Send request to model
            response = requests.post(
                "http://localhost:8080/api/chat", #todo: check that the endpoint is correct!
                headers={"Content-Type": "application/json"},
                json={
                    "model": "mistral-nemo",
                    "stream": False,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": 0.5
                },
            )
            
            if response.status_code != 200:
                logger.error(f"Error from LLM API: {response.text}")
                return jsonify({'error': f'LLM API error: {response.text}'}), 500
            
            try:
                data = json.loads(response.text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                logger.info(f"Raw response text: {response.text}")
                return jsonify({'error': f'Invalid JSON from LLM: {str(e)}'}), 500
            #logger.info(f"Raw response text: {response.text}")
            logger.info(f"Raw data!!!!!!!! {data}")
            assistant_message = data['message']
            messages.append(assistant_message)
            
            
            logger.info(f"Assistant message: {assistant_message}")
            
            # Check if tool call exists
            if "tool_calls" in assistant_message:
                executed_tools = False
                for tool_call in assistant_message["tool_calls"]:
                    # Extract tool info
                    function_name = tool_call["function"]["name"]
                    function_args = json.loads(tool_call["function"]["arguments"])
                    
                    # Execute the function
                    if function_name in tool_map:
                        executed_tools = True
                        result = tool_map[function_name](**function_args)
                        
                        # Add result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call["id"],
                            "name": function_name,
                            "content": str(result)
                        })
                
                # If no tools were executed, break the loop
                if not executed_tools:
                    break
            else:
                # No more tool calls, we're done
                break
        
        # Get the final response
        final_response = assistant_message["content"] if "content" in assistant_message else "Calculation completed"
        
        return jsonify({'response': final_response})
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500
    
if __name__ == '__main__':
    CORS(app)
    logger.info("LP Agent API starting up...")
    app.run(host='0.0.0.0', port=5001, debug=False)