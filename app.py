# Save this file as `app.py`

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteReadTool

# --- INITIAL SETUP ---
# Load environment variables from a .env file (for your GEMINI_API_KEY)
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS (Cross-Origin Resource Sharing) to allow your WordPress site
# to communicate with this API from a different domain.
CORS(app)

# --- CREWAI AGENT & TOOL SETUP ---
# It's good practice to set up tools and agents outside the request function
# so they don't get re-initialized on every single API call, saving resources.
try:
    search_tool = SerperDevTool()
    website_tool = WebsiteReadTool()

    # Agent 1: The News Scout
    news_scout = Agent(
      role='Lead Market Researcher',
      goal='Find the latest news, press releases, and blog posts for a given company.',
      backstory="""You are an expert market researcher, skilled at using web search tools
      to find the most relevant, recent, and impactful news and official announcements
      from a company's website and top-tier news outlets.""",
      verbose=False,
      allow_delegation=False,
      tools=[search_tool, website_tool]
    )

    # Agent 2: The Social Media Monitor
    social_monitor = Agent(
      role='Social Media Analyst',
      goal='Analyze the social media presence and customer sentiment of a given company.',
      backstory="""You are a seasoned social media analyst. You know how to dig through
      social platforms and forums like Reddit to gauge public opinion, identify marketing
      campaigns, and find out what real people are saying.""",
      verbose=False,
      allow_delegation=False,
      tools=[search_tool]
    )

    # Agent 3: The Strategy Analyst (with Freemium Logic)
    strategy_analyst = Agent(
      role='Senior Business Strategist',
      goal='Synthesize research findings into a comprehensive, actionable intelligence briefing, tailored to the user\'s subscription plan.',
      backstory="""You are a Senior Business Strategist with a talent for creating compelling reports from raw data.
      For 'free' users, you provide a tantalizing summary to encourage upgrades.
      For 'paid' users, you deliver the full, in-depth analysis.""",
      verbose=False,
      allow_delegation=False,
    )
except Exception as e:
    print(f"Error setting up agents or tools: {e}")

# --- API ENDPOINT DEFINITION ---
# This defines the URL endpoint for our API.
# The endpoint will be '/generate-report' and will only accept POST requests.
@app.route('/generate-report', methods=['POST'])
def generate_report_endpoint():
    """
    This function is triggered when a request is made to the /generate-report endpoint.
    It runs the CrewAI process based on the provided company name and plan type,
    then returns the generated report.
    """
    # First, check if the Gemini API key is available in the environment.
    if not os.getenv('GEMINI_API_KEY'):
        return jsonify({"error": "Gemini API key not found. Please set it in the environment variables."}), 500

    # Get the JSON data from the incoming request.
    data = request.get_json()
    if not data or 'company_name' not in data:
        return jsonify({"error": "Missing 'company_name' in request body"}), 400

    competitor_company = data['company_name']
    # Default to the 'free' plan if 'plan_type' is not specified in the request.
    # This is crucial for our new freemium model.
    plan_type = data.get('plan_type', 'free')

    # --- DYNAMIC TASK & CREW CREATION ---
    # We create the tasks and crew inside this function to use the company name
    # and plan type provided in each unique API request.

    news_task = Task(
      description=f'Search for and analyze the latest news, blog posts, and press releases for the company: {competitor_company}. Focus on the last 3-6 months. Summarize key product launches, leadership changes, and financial news.',
      expected_output='A bullet-point summary of the top 3-5 most important news items and announcements.',
      agent=news_scout
    )

    social_task = Task(
      description=f'Analyze the social media presence of {competitor_company}. Search for discussions on Reddit and Twitter. Identify their main marketing messages and summarize the overall public sentiment (positive, negative, neutral).',
      expected_output='A summary of the company\'s social media strategy and a paragraph describing customer sentiment with examples.',
      agent=social_monitor
    )

    # --- NEW: Plan-based Strategy Task ---
    # The description for the final report now changes based on the user's plan.
    if plan_type == 'paid':
        strategy_task_description = f"""
        Using the provided news analysis and social media sentiment for {competitor_company}, compile a COMPLETE and IN-DEPTH competitor intelligence report.
        The report MUST have the following sections, each fully detailed:
        1.  **Key Announcements:** Full details of recent news.
        2.  **Social Media & Customer Voice:** Deep dive into their strategy and customer feedback.
        3.  **SWOT Analysis:** A full Strengths, Weaknesses, Opportunities, and Threats analysis based on all data.
        4.  **Content Strategy Angles:** Suggest 3-5 specific content ideas or keywords the user can target.
        """
        expected_output_format = 'A well-formatted, professional intelligence report using markdown. Provide full, unrestricted details in all sections.'
    else: # This is the 'free' plan
        strategy_task_description = f"""
        Using the provided news analysis and social media sentiment for {competitor_company}, compile a "teaser" competitor intelligence report.
        The report MUST have the following structure:
        1.  **Key Announcements:** Provide the headlines of the top 2 news items, but only a one-sentence summary for each.
        2.  **Social Media & Customer Voice:** Give a one-sentence overview of their social strategy and the general sentiment (e.g., "Generally positive").
        3.  **Premium Features Locked:** Create sections titled "SWOT Analysis" and "Content Strategy Angles" but under each, write: 'Upgrade to a paid plan to unlock this in-depth analysis.'
        """
        expected_output_format = 'A well-formatted "teaser" report using markdown. It should give a taste of the value while clearly indicating that more is available in the paid version.'

    strategy_task = Task(
      description=strategy_task_description,
      expected_output=expected_output_format,
      agent=strategy_analyst,
      context=[news_task, social_task]
    )

    # Assemble the crew with the dynamically defined strategy task
    competitor_crew = Crew(
      agents=[news_scout, social_monitor, strategy_analyst],
      tasks=[news_task, social_task, strategy_task],
      process=Process.sequential,
      verbose=False
    )

    # Kick off the crew's work and handle potential errors
    try:
        report = competitor_crew.kickoff()
        # Return the final report as a JSON response with a 200 OK status
        return jsonify({"report": report})
    except Exception as e:
        # Return a clear error message if something goes wrong during the kickoff
        return jsonify({"error": f"An error occurred while generating the report: {str(e)}"}), 500

# This part allows the script to be run directly for local testing.
# When deployed on Render, Gunicorn will handle this.
if __name__ == '__main__':
    # The app will run on http://127.0.0.1:5000 when tested locally
    app.run(debug=True, port=5000)
