"""
Multi-Agent AI Financial Analyst

A Streamlit application that uses AI agents to analyze stocks and generate
comprehensive financial reports.
"""

import streamlit as st
import os
import pandas as pd
import re
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from pydantic import BaseModel, validator
from tools.financial_tools import YFinanceStockTool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_STOCK_SYMBOL = "AAPL"
MAX_RETRIES = 3
API_TIMEOUT = 30
VALID_STOCK_PATTERN = re.compile(r'^[A-Z]{1,5}$')

class StockAnalysis(BaseModel):
    """Pydantic model for structured stock analysis output."""
    symbol: str
    company_name: str
    current_price: float
    market_cap: float
    pe_ratio: Optional[float] = None
    recommendation: str
    analysis_summary: str
    risk_assessment: str
    technical_indicators: Dict[str, Any]
    fundamental_metrics: Dict[str, Any]
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate stock symbol format."""
        if not VALID_STOCK_PATTERN.match(v.upper()):
            raise ValueError('Invalid stock symbol format')
        return v.upper()

class FinancialAnalystApp:
    """Main application class for the Financial Analyst."""
    
    def __init__(self):
        self.llm = None
        self._initialize_session_state()
    
    def _initialize_session_state(self) -> None:
        """Initialize Streamlit session state variables."""
        if "analysis_complete" not in st.session_state:
            st.session_state.analysis_complete = False
        if "report" not in st.session_state:
            st.session_state.report = None
        if "api_key_validated" not in st.session_state:
            st.session_state.api_key_validated = False
    
    @st.cache_resource
    def _load_llm(_self, api_key: str) -> LLM:
        """Initialize and cache SambaNova LLM instance."""
        try:
            return LLM(
                model="sambanova/Llama-4-Maverick-17B-128E-Instruct",
                api_key=api_key,
                temperature=0.3
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and availability."""
        if not api_key or len(api_key.strip()) < 10:
            return False
        # Add more sophisticated validation if needed
        return True
    
    def _validate_stock_symbol(self, symbol: str) -> bool:
        """Validate stock symbol format."""
        return bool(VALID_STOCK_PATTERN.match(symbol.upper()))
    
    def _create_analysis_agent(self, symbol: str, llm: LLM) -> Agent:
        """Create stock analysis agent."""
        stock_tool = YFinanceStockTool()
        
        return Agent(
            role="Wall Street Financial Analyst",
            goal=f"Conduct comprehensive analysis of {symbol} using real-time data",
            backstory="""
            Seasoned Wall Street analyst with 15+ years of experience in equity research.
            Known for meticulous analysis and data-driven insights.
            Expert at interpreting financial metrics and providing actionable insights.
            """,
            llm=llm,
            verbose=True,
            memory=True,
            tools=[stock_tool]
        )
    
    def _create_report_agent(self, llm: LLM) -> Agent:
        """Create report writing agent."""
        return Agent(
            role="Financial Report Specialist",
            goal="Transform analysis into professional investment report",
            backstory="""
            Expert financial writer creating institutional-grade research reports.
            Excel at presenting complex data in clear, structured format.
            Known for clear data presentation and risk assessment capabilities.
            """,
            llm=llm,
            verbose=True
        )
    
    def _create_analysis_task(self, symbol: str, agent: Agent) -> Task:
        """Create analysis task with detailed requirements."""
        description = f"""
        Analyze {symbol} stock using stock_data_tool for real-time data.
        
        Required analysis components:
        1. Latest Trading Information (HIGHEST PRIORITY)
        2. 52-Week Performance (CRITICAL)
        3. Financial Deep Dive
        4. Technical Analysis
        5. Market Context
        
        IMPORTANT: Always use real-time data and include specific dates.
        """
        
        return Task(
            description=description,
            expected_output="Comprehensive analysis with real-time data and metrics",
            agent=agent
        )
    
    def _create_report_task(self, symbol: str, agent: Agent) -> Task:
        """Create report generation task."""
        description = f"""
        Transform analysis into professional investment report for {symbol}.
        
        Required sections:
        - Executive Summary
        - Market Position Overview
        - Financial Metrics Analysis
        - Technical Analysis
        - Risk Assessment
        - Future Outlook
        
        Format: Clean markdown with tables and visual indicators.
        """
        
        return Task(
            description=description,
            expected_output="Professional markdown-formatted investment report",
            agent=agent
        )
    
    def create_crew(self, symbol: str, api_key: str) -> Crew:
        """Create and configure the analysis crew."""
        try:
            llm = self._load_llm(api_key)
            
            # Create agents
            analysis_agent = self._create_analysis_agent(symbol, llm)
            report_agent = self._create_report_agent(llm)
            
            # Create tasks
            analysis_task = self._create_analysis_task(symbol, analysis_agent)
            report_task = self._create_report_task(symbol, report_agent)
            
            return Crew(
                agents=[analysis_agent, report_agent],
                tasks=[analysis_task, report_task],
                process=Process.sequential,
                verbose=True
            )
        except Exception as e:
            logger.error(f"Failed to create crew: {e}")
            raise
    
    def run_analysis(self, symbol: str, api_key: str) -> str:
        """Run the stock analysis with error handling and retries."""
        for attempt in range(MAX_RETRIES):
            try:
                crew = self.create_crew(symbol, api_key)
                result = crew.kickoff()
                
                # Handle different result types
                if hasattr(result, 'raw'):
                    return result.raw
                return str(result)
                
            except Exception as e:
                logger.warning(f"Analysis attempt {attempt + 1} failed: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                continue
    
    def render_sidebar(self) -> tuple[str, str, bool]:
        """Render sidebar and return user inputs."""
        with st.sidebar:
            st.header("Configuration")
            
            # API Key input with validation
            api_key = st.text_input(
                "SambaNova API Key",
                type="password",
                value=os.getenv("SAMBANOVA_API_KEY", ""),
                help="Enter your SambaNova API key"
            )
            
            # Validate API key
            if api_key and not self._validate_api_key(api_key):
                st.error("Invalid API key format")
            
            # Stock Symbol input with validation
            symbol = st.text_input(
                "Stock Symbol",
                value=DEFAULT_STOCK_SYMBOL,
                help="Enter a stock symbol (e.g., AAPL, GOOGL)"
            ).upper()
            
            # Validate stock symbol
            if symbol and not self._validate_stock_symbol(symbol):
                st.error("Invalid stock symbol format (1-5 uppercase letters)")
            
            # Analysis button
            analyze_button = st.button(
                "Analyze Stock", 
                type="primary",
                disabled=not (api_key and symbol and 
                            self._validate_api_key(api_key) and 
                            self._validate_stock_symbol(symbol))
            )
            
            return api_key, symbol, analyze_button
    
    def render_main_content(self, symbol: str) -> None:
        """Render main content area."""
        if st.session_state.analysis_complete and st.session_state.report:
            st.markdown("### Analysis Report")
            st.markdown(st.session_state.report)
            
            # Download button
            st.download_button(
                label="Download Report",
                data=st.session_state.report,
                file_name=f"stock_analysis_{symbol}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    def run(self) -> None:
        """Main application entry point."""
        st.set_page_config(
            page_title="Multi-Agent AI Financial Analyst", 
            layout="wide"
        )
        
        st.title("🎯 Multi-Agent AI Financial Analyst")
        
        # Render sidebar
        api_key, symbol, analyze_button = self.render_sidebar()
        
        # Handle analysis request
        if analyze_button:
            try:
                with st.spinner(f'Analyzing {symbol}... This may take a few minutes.'):
                    report = self.run_analysis(symbol, api_key)
                    st.session_state.report = report
                    st.session_state.analysis_complete = True
                    st.success("Analysis completed successfully!")
                    
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                st.error(f"Analysis failed: {str(e)}")
                st.session_state.analysis_complete = False
        
        # Render main content
        self.render_main_content(symbol)
        
        # Footer
        st.markdown("---")
        st.markdown("*Powered by SambaNova AI and CrewAI*")

def main():
    """Application entry point."""
    app = FinancialAnalystApp()
    app.run()

if __name__ == "__main__":
    main()