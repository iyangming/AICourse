"""
多智能体AI金融分析师

一个使用AI智能体分析股票并生成综合金融报告的Streamlit应用程序。
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

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

# 常量定义
DEFAULT_STOCK_SYMBOL = "AAPL"  # 默认股票代码
MAX_RETRIES = 3  # 最大重试次数
API_TIMEOUT = 30  # API超时时间（秒）
VALID_STOCK_PATTERN = re.compile(r'^[A-Z]{1,5}$')  # 有效股票代码格式

class StockAnalysis(BaseModel):
    """股票分析结构化输出的Pydantic模型。"""
    symbol: str  # 股票代码
    company_name: str  # 公司名称
    current_price: float  # 当前价格
    market_cap: float  # 市值
    pe_ratio: Optional[float] = None  # 市盈率（可选）
    recommendation: str  # 投资建议
    analysis_summary: str  # 分析摘要
    risk_assessment: str  # 风险评估
    technical_indicators: Dict[str, Any]  # 技术指标
    fundamental_metrics: Dict[str, Any]  # 基本面指标
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """验证股票代码格式。"""
        if not VALID_STOCK_PATTERN.match(v.upper()):
            raise ValueError('无效的股票代码格式')
        return v.upper()

class FinancialAnalystApp:
    """金融分析师应用程序的主类。"""
    
    def __init__(self):
        """初始化应用程序。"""
        self.llm = None  # LLM实例
        self._initialize_session_state()  # 初始化会话状态
    
    def _initialize_session_state(self) -> None:
        """初始化Streamlit会话状态变量。"""
        if "analysis_complete" not in st.session_state:
            st.session_state.analysis_complete = False  # 分析完成标志
        if "report" not in st.session_state:
            st.session_state.report = None  # 分析报告
        if "api_key_validated" not in st.session_state:
            st.session_state.api_key_validated = False  # API密钥验证状态
    
    @st.cache_resource
    def _load_llm(_self, api_key: str) -> LLM:
        """初始化并缓存SambaNova LLM实例。"""
        try:
            return LLM(
                model="sambanova/Llama-4-Maverick-17B-128E-Instruct",
                api_key=api_key,
                temperature=0.3  # 控制输出的随机性
            )
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            raise
    
    def _validate_api_key(self, api_key: str) -> bool:
        """验证API密钥格式和可用性。"""
        if not api_key or len(api_key.strip()) < 10:
            return False
        # 如需要可添加更复杂的验证逻辑
        return True
    
    def _validate_stock_symbol(self, symbol: str) -> bool:
        """验证股票代码格式。"""
        return bool(VALID_STOCK_PATTERN.match(symbol.upper()))
    
    def _create_analysis_agent(self, symbol: str, llm: LLM) -> Agent:
        """创建股票分析智能体。"""
        stock_tool = YFinanceStockTool()  # 初始化股票数据工具
        
        return Agent(
            role="华尔街金融分析师",
            goal=f"使用实时数据对{symbol}进行全面分析",
            backstory="""
            拥有15年以上股票研究经验的资深华尔街分析师。
            以细致的分析和数据驱动的洞察而闻名。
            擅长解读财务指标并提供可操作的投资建议。
            """,
            llm=llm,
            verbose=True,  # 启用详细输出
            memory=True,   # 启用记忆功能
            tools=[stock_tool]  # 配置可用工具
        )
    
    def _create_report_agent(self, llm: LLM) -> Agent:
        """创建报告撰写智能体。"""
        return Agent(
            role="金融报告专家",
            goal="将分析转换为专业的投资报告",
            backstory="""
            专业的金融写作专家，创建机构级研究报告。
            擅长以清晰、结构化的格式呈现复杂数据。
            以清晰的数据展示和风险评估能力而著称。
            """,
            llm=llm,
            verbose=True
        )
    
    def _create_analysis_task(self, symbol: str, agent: Agent) -> Task:
        """创建具有详细要求的分析任务。"""
        description = f"""
        使用stock_data_tool获取实时数据分析{symbol}股票。
        
        必需的分析组件：
        1. 最新交易信息（最高优先级）
        2. 52周表现（关键）
        3. 财务深度分析
        4. 技术分析
        5. 市场背景
        
        重要提示：始终使用实时数据并包含具体日期。
        """
        
        return Task(
            description=description,
            expected_output="包含实时数据和指标的综合分析",
            agent=agent
        )
    
    def _create_report_task(self, symbol: str, agent: Agent) -> Task:
        """创建报告生成任务。"""
        description = f"""
        将{symbol}的分析转换为专业投资报告。
        
        必需章节：
        - 执行摘要
        - 市场地位概述
        - 财务指标分析
        - 技术分析
        - 风险评估
        - 未来展望
        
        格式：清晰的markdown格式，包含表格和视觉指示器。
        """
        
        return Task(
            description=description,
            expected_output="专业的markdown格式投资报告",
            agent=agent
        )
    
    def create_crew(self, symbol: str, api_key: str) -> Crew:
        """创建和配置分析团队。"""
        try:
            llm = self._load_llm(api_key)  # 加载LLM
            
            # 创建智能体
            analysis_agent = self._create_analysis_agent(symbol, llm)
            report_agent = self._create_report_agent(llm)
            
            # 创建任务
            analysis_task = self._create_analysis_task(symbol, analysis_agent)
            report_task = self._create_report_task(symbol, report_agent)
            
            return Crew(
                agents=[analysis_agent, report_agent],
                tasks=[analysis_task, report_task],
                process=Process.sequential,  # 顺序执行
                verbose=True
            )
        except Exception as e:
            logger.error(f"团队创建失败: {e}")
            raise
    
    def run_analysis(self, symbol: str, api_key: str) -> str:
        """运行股票分析，包含错误处理和重试机制。"""
        for attempt in range(MAX_RETRIES):
            try:
                crew = self.create_crew(symbol, api_key)  # 创建分析团队
                result = crew.kickoff()  # 启动分析
                
                # 处理不同的结果类型
                if hasattr(result, 'raw'):
                    return result.raw
                return str(result)
                
            except Exception as e:
                logger.warning(f"分析尝试 {attempt + 1} 失败: {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                continue
    
    def render_sidebar(self) -> tuple[str, str, bool]:
        """渲染侧边栏并返回用户输入。"""
        with st.sidebar:
            st.header("配置")
            
            # API密钥输入和验证
            api_key = st.text_input(
                "SambaNova API密钥",
                type="password",
                value=os.getenv("SAMBANOVA_API_KEY", ""),
                help="请输入您的SambaNova API密钥"
            )
            
            # 验证API密钥
            if api_key and not self._validate_api_key(api_key):
                st.error("无效的API密钥格式")
            
            # 股票代码输入和验证
            symbol = st.text_input(
                "股票代码",
                value=DEFAULT_STOCK_SYMBOL,
                help="请输入股票代码（例如：AAPL, GOOGL）"
            ).upper()
            
            # 验证股票代码
            if symbol and not self._validate_stock_symbol(symbol):
                st.error("无效的股票代码格式（1-5个大写字母）")
            
            # 分析按钮
            analyze_button = st.button(
                "分析股票", 
                type="primary",
                disabled=not (api_key and symbol and 
                            self._validate_api_key(api_key) and 
                            self._validate_stock_symbol(symbol))
            )
            
            return api_key, symbol, analyze_button
    
    def render_main_content(self, symbol: str) -> None:
        """渲染主内容区域。"""
        if st.session_state.analysis_complete and st.session_state.report:
            st.markdown("### 分析报告")
            st.markdown(st.session_state.report)
            
            # 下载按钮
            st.download_button(
                label="下载报告",
                data=st.session_state.report,
                file_name=f"stock_analysis_{symbol}_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
    
    def run(self) -> None:
        """应用程序主入口点。"""
        st.set_page_config(
            page_title="多智能体AI金融分析师", 
            layout="wide"
        )
        
        st.title("🎯 多智能体AI金融分析师")
        
        # 渲染侧边栏
        api_key, symbol, analyze_button = self.render_sidebar()
        
        # 处理分析请求
        if analyze_button:
            try:
                with st.spinner(f'正在分析 {symbol}... 这可能需要几分钟时间。'):
                    report = self.run_analysis(symbol, api_key)
                    st.session_state.report = report
                    st.session_state.analysis_complete = True
                    st.success("分析完成！")
                    
            except Exception as e:
                logger.error(f"分析失败: {e}")
                st.error(f"分析失败: {str(e)}")
                st.session_state.analysis_complete = False
        
        # 渲染主内容
        self.render_main_content(symbol)
        
        # 页脚
        st.markdown("---")
        st.markdown("*由SambaNova AI和CrewAI提供支持*")

def main():
    """应用程序入口点。"""
    app = FinancialAnalystApp()
    app.run()

if __name__ == "__main__":
    main()