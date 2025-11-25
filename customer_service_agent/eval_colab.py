{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Evaluate ADK Customer Service Agent on Vertex AI\n",
    "\n",
    "This notebook demonstrates how to evaluate an ADK-based customer service agent using Vertex AI Gen AI Evaluation.\n",
    "\n",
    "## Setup\n",
    "\n",
    "Install necessary packages."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q \"google-cloud-aiplatform[agent_engines,evaluation]\" google-adk python-dotenv pandas plotly"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Authentication\n",
    "\n",
    "Authenticate to Google Cloud."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "if 'google.colab' in sys.modules:\n",
    "    from google.colab import auth\n",
    "    auth.authenticate_user()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Initialize Vertex AI\n",
    "\n",
    "Set your Project ID and Location."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import vertexai\n",
    "import os\n",
    "\n",
    "PROJECT_ID = \"[your-project-id]\"  # @param {type:\"string\"}\n",
    "LOCATION = \"us-central1\"  # @param {type:\"string\"}\n",
    "\n",
    "if PROJECT_ID == \"[your-project-id]\":\n",
    "    print(\"Please set your Project ID\")\n",
    "else:\n",
    "    vertexai.init(project=PROJECT_ID, location=LOCATION)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Define ADK Agent\n",
    "\n",
    "We define the agent and its tools, similar to `agent.py`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from typing import Dict, Any\n",
    "from google.adk.agents import Agent\n",
    "from google.adk.sessions import InMemorySessionService\n",
    "\n",
    "# Mock data\n",
    "MOCK_DATA = {\n",
    "    \"CUST001\": {\n",
    "        \"orders\": [\n",
    "            {\"order_id\": \"ORD-101\", \"date\": \"2023-10-15\", \"items\": [\"Wireless Headphones\"], \"total\": 120.00, \"status\": \"delivered\"},\n",
    "            {\"order_id\": \"ORD-102\", \"date\": \"2023-11-01\", \"items\": [\"USB-C Cable\", \"Phone Case\"], \"total\": 35.00, \"status\": \"shipped\"}\n",
    "        ]\n",
    "    },\n",
    "    \"CUST002\": {\n",
    "        \"orders\": [\n",
    "            {\"order_id\": \"ORD-201\", \"date\": \"2023-09-20\", \"items\": [\"Smart Watch\"], \"total\": 250.00, \"status\": \"delivered\"}\n",
    "        ]\n",
    "    }\n",
    "}\n",
    "\n",
    "def get_purchase_history(customer_id: str) -> Dict[str, Any]:\n",
    "    \"\"\"Retrieves the purchase history for a given customer.\"\"\"\n",
    "    return MOCK_DATA.get(customer_id, {\"orders\": [], \"message\": \"No purchase history found.\"})\n",
    "\n",
    "def issue_refund(order_id: str, reason: str) -> Dict[str, Any]:\n",
    "    \"\"\"Issues a refund for a specific order.\"\"\"\n",
    "    return {\"status\": \"success\", \"order_id\": order_id, \"message\": f\"Refund processed for {reason}\"}\n",
    "\n",
    "def lookup_product_info(product_name: str) -> Dict[str, Any]:\n",
    "    \"\"\"Looks up details for a specific product.\"\"\"\n",
    "    products = {\n",
    "        \"wireless headphones\": {\"price\": 120.00, \"in_stock\": True, \"description\": \"Noise-canceling\"},\n",
    "        \"smart watch\": {\"price\": 250.00, \"in_stock\": False, \"description\": \"Fitness tracking\"}\n",
    "    }\n",
    "    return products.get(product_name.lower(), {\"message\": \"Product not found.\"})\n",
    "\n",
    "agent_instruction = \"\"\"\n",
    "You are a helpful retail customer service representative.\n",
    "Use tools to assist with purchase history, refunds, and product inquiries.\n",
    "\"\"\"\n",
    "\n",
    "agent = Agent(\n",
    "    model=\"gemini-1.5-flash\", # Using a stable model for Colab\n",
    "    name=\"customer_service_agent\",\n",
    "    instruction=agent_instruction,\n",
    "    tools=[get_purchase_history, issue_refund, lookup_product_info],\n",
    ")\n",
    "\n",
    "session_service = InMemorySessionService()\n",
    "\n",
    "def run_agent(input_text: str) -> Dict[str, Any]:\n",
    "    session = session_service.get_session(app_name=\"eval_app\", user_id=\"eval_user\")\n",
    "    response = session.send_message(input_text)\n",
    "    \n",
    "    # Extract trajectory (tool calls)\n",
    "    trajectory = []\n",
    "    # Note: ADK's response object structure might vary, \n",
    "    # here we assume we can access tool calls if any.\n",
    "    # For simplicity in this demo, we focus on the final response.\n",
    "    # To get trajectory, we'd need to inspect session events.\n",
    "    \n",
    "    return {\n",
    "        \"response\": response.text,\n",
    "        \"predicted_trajectory\": trajectory # ADK trajectory extraction needs more work for full evaluation\n",
    "    }"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Prepare Evaluation Dataset\n",
    "\n",
    "We convert the evaluation cases from `eval.test.json` into a format suitable for Vertex AI Evaluation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "\n",
    "eval_data = {\n",
    "    \"prompt\": [\n",
    "        \"Do you have wireless headphones in stock?\",\n",
    "        \"What did I buy recently? My customer ID is CUST001.\",\n",
    "        \"I want a refund for order ORD-102 because it was damaged.\"\n",
    "    ],\n",
    "    \"reference_response\": [\n",
    "        \"Yes, we have wireless headphones in stock! They are priced at $120.00.\",\n",
    "        \"Here's your recent purchase history for Customer ID CUST001...\",\n",
    "        \"Your refund for order ORD-102 has been processed.\"\n",
    "    ],\n",
    "    # For trajectory evaluation, we would add reference_trajectory here\n",
    "}\n",
    "\n",
    "eval_df = pd.DataFrame(eval_data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Run Evaluation\n",
    "\n",
    "We evaluate the agent's responses using coherence and safety metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from vertexai.preview.evaluation import EvalTask\n",
    "\n",
    "def agent_fn(input_text: str) -> str:\n",
    "    return run_agent(input_text)[\"response\"]\n",
    "\n",
    "eval_task = EvalTask(\n",
    "    dataset=eval_df,\n",
    "    metrics=[\"coherence\", \"safety\"],\n",
    "    experiment=\"adk-agent-eval\"\n",
    ")\n",
    "\n",
    "result = eval_task.evaluate(runnable=agent_fn)\n",
    "\n",
    "print(\"Summary Metrics:\")\n",
    "print(result.summary_metrics)\n",
    "print(\"\\nRow-wise Metrics:\")\n",
    "print(result.metrics_table)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
