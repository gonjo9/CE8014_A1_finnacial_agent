import json
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

""" Gemini API client """
# MODEL_NAME = "gemini-2.0-flash"
# client = OpenAI(
#     api_key=os.getenv("GEMINI_API_KEY"),
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )

""" local LLM client (ollama) """
MODEL_NAME = "qwen2.5:3b"
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)


# Mock Functions
def get_exchange_rate(currency_pair: str):
    """Mock function to get exchange rate."""
    print(f"[Mock] Getting exchange rate for {currency_pair}...")
    data = {
        "USD_TWD": "32.0",
        "JPY_TWD": "0.2",
        "EUR_USD": "1.2",
    }
    if currency_pair in data:
        return json.dumps({"currency_pair": currency_pair, "rate": data[currency_pair]})
    return json.dumps({"error": "Data not found"})

def get_stock_price(symbol: str):
    """Mock function to get stock price."""
    print(f"[Mock] Getting stock price for {symbol}...")
    data = {
        "AAPL": "260.00",
        "TSLA": "430.00",
        "NVDA": "190.00",
    }
    if symbol in data:
        return json.dumps({"symbol": symbol, "price": data[symbol]})
    return json.dumps({"error": "Data not found"})

available_functions = {
    "get_exchange_rate": get_exchange_rate,
    "get_stock_price": get_stock_price,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get exchange rate for a given currency pair",
            "parameters": {
                "type": "object",
                "properties": {
                    "currency_pair": {"type": "string", "description": "Currency pair (e.g. USD_EUR)"}
                },
                "required": ["currency_pair"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get stock price for a given symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock symbol (e.g. AAPL)"}
                },
                "required": ["symbol"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
]

def run_agent():
    messages = [
        {
            "role": "system",
            "content": 
            """
            You are a helpful financial assistant. Use tools when the user asks for stock prices or exchange rates.
            """
        }
    ]

    print("Agent Started. Type 'exit' to quit.")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        messages.append({"role": "user", "content": user_input})

        # First API Call
        response = client.chat.completions.create(
            # model="gpt-4o-mini",
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_msg = response.choices[0].message
        tool_calls = response_msg.tool_calls

        if tool_calls:
            # IMPORTANT: Add the assistant's "thought" (tool call request) to history
            messages.append(response_msg)
            
            # 6. Handle Parallel Tool Calls
            # The model might call multiple tools in one go (e.g. "Time in Taipei and NY")
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Dynamic Dispatch using Function Map
                function_to_call = available_functions.get(function_name)
                
                if function_to_call:
                    try:
                        tool_result = function_to_call(**function_args)
                    except Exception as e:
                        tool_result = json.dumps({"error": str(e)})
                else:
                    tool_result = json.dumps({"error": "Function not found"})
                
                # Append RESULT to history
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": tool_result,
                })
            
            # Second API Call: nudge local LLM to use tool results (not added to history)
            synthesis_prompt = "Based on the tool results above, give a short reply. Use the exact numbers from each tool result (price and rate). Do not say data is unavailable if a result already contains a number."
            final_response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages + [{"role": "user", "content": synthesis_prompt}]
            )
            final_content = final_response.choices[0].message.content
            print(f"Agent: {final_content}")
            messages.append({"role": "assistant", "content": final_content})
            
        else:
            # No tool needed
            print(f"Agent: {response_msg.content}")
            messages.append({"role": "assistant", "content": response_msg.content})

if __name__ == "__main__":
    run_agent()