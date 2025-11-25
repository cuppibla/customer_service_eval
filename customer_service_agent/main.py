import asyncio
from google.adk.agents import LlmAgent
from agent import root_agent

async def run_agent_query(query: str):
    print(f"\n--- Query: {query} ---")
    # In a real scenario, you might use a session to maintain state, 
    # but for a simple test, we can run a single query.
    # For ADK, we typically use a runner or session, but for a quick check:
    from google.adk.runtime import Session
    from google.adk.services import InMemorySessionService
    
    session_service = InMemorySessionService()
    session = await session_service.create_session()
    
    response = await root_agent.run(query, session=session)
    print(f"Response: {response.text}")

async def main():
    # Test case 1: Product info
    await run_agent_query("Do you have wireless headphones in stock?")
    
    # Test case 2: Purchase history (needs customer ID)
    await run_agent_query("What did I buy recently? My customer ID is CUST001.")
    
    # Test case 3: Refund (needs order ID and reason)
    await run_agent_query("I want a refund for order ORD-102 because it was damaged.")

if __name__ == "__main__":
    asyncio.run(main())
