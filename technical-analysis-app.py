import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import anthropic
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class CryptoAnalysisApp:
    def __init__(self):
        # Use Streamlit secrets instead of .env file
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        if not api_key:
            raise ValueError("No API key found in secrets")
        self.client = anthropic.Client(api_key=api_key)
        # Common cryptocurrency symbols
        
        self.crypto_symbols = {
            'Bitcoin (BTC)': 'BTC-USD',
            'Ethereum (ETH)': 'ETH-USD',
            'Solana (SOL)': 'SOL-USD',
            'Cardano (ADA)': 'ADA-USD',
            'Ripple (XRP)': 'XRP-USD',
            'Dogecoin (DOGE)': 'DOGE-USD',
            'Polygon (MATIC)': 'MATIC-USD',
            'Polkadot (DOT)': 'DOT-USD',
            'Avalanche (AVAX)': 'AVAX-USD',
            'Chainlink (LINK)': 'LINK-USD',
            'Uniswap (UNI)': 'UNI-USD',
            'Binance Coin (BNB)': 'BNB-USD',
        }
    
    def calculate_indicators(self, df):
        """Calculate technical indicators specific to crypto"""
        # Traditional indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # Crypto-specific: Volume Moving Average
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Volatility (Daily Returns)
        df['Daily_Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(365)
        
        return df
    
    def create_analysis_plots(self, df):
        """Create technical analysis plots"""
        fig = make_subplots(
            rows=4,  # Added an extra row for volume
            cols=1,
            subplot_titles=('Price and Moving Averages', 'Volume', 'MACD', 'RSI'),
            row_heights=[0.4, 0.2, 0.2, 0.2],
            shared_xaxes=True,
            vertical_spacing=0.05
        )

        # Price and MA
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, 
            col=1
        )
        
        # Add Moving Averages
        for ma, color in [('SMA_20', 'orange'), ('SMA_50', 'blue')]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[ma],
                    name=ma,
                    line=dict(color=color)
                ),
                row=1,
                col=1
            )

        # Volume
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color='purple'
            ),
            row=2,
            col=1
        )

        # MACD
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=3,
            col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Signal_Line'],
                name='Signal Line',
                line=dict(color='orange')
            ),
            row=3,
            col=1
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='purple')
            ),
            row=4,
            col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

        fig.update_layout(
            height=1000,  # Increased height for better visualization
            title_text="Cryptocurrency Technical Analysis",
            xaxis_rangeslider_visible=False,
            showlegend=True
        )

        # Update y-axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=4, col=1)

        return fig

    def get_analysis(self, symbol, timeframe='6mo'):
        """Generate analysis for the given symbol"""
        try:
            # Get data using the USD pair
            crypto = yf.Ticker(symbol)
            df = crypto.history(period=timeframe)
            if df.empty:
                return None, "No data found for this cryptocurrency", None
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            
            # Create plots
            fig = self.create_analysis_plots(df)
            
            # Get current indicators
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Prepare crypto-specific analysis prompt
            analysis_prompt = f"""
            Analyze the following technical indicators for {symbol}:
            
            Current Price: ${current['Close']:.2f}
            24h Volume: {current['Volume']:,.0f}
            
            Technical Indicators:
            - SMA 20: ${current['SMA_20']:.2f}
            - SMA 50: ${current['SMA_50']:.2f}
            - MACD: {current['MACD']:.3f}
            - Signal Line: {current['Signal_Line']:.3f}
            - RSI: {current['RSI']:.2f}
            - 20-day Volatility: {current['Volatility']*100:.1f}%
            
            Please provide a cryptocurrency-specific analysis including:
            1. Overall trading recommendation (Buy/Sell/Hold)
            2. Key technical signals and market sentiment
            3. Volume analysis and liquidity conditions
            4. Volatility assessment
            5. Key support and resistance levels
            6. Risk factors specific to cryptocurrency markets
            
            Format the response in clear sections with bullet points.
            """
            
            response = self.client.messages.create(
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": analysis_prompt
                }],
                model="claude-3-opus-20240229"
            )
            
            return df, response.content[0].text, fig
            
        except Exception as e:
            return None, f"Error: {str(e)}", None

def main():
    st.set_page_config(page_title="Crypto Technical Analysis", layout="wide")
    
    st.title("🚀 Cryptocurrency Technical Analysis Assistant")
    
    try:
        analyzer = CryptoAnalysisApp()
    except ValueError as e:
        st.error(f"Configuration error: {str(e)}")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        # Create a dropdown for cryptocurrency selection
        selected_crypto = st.selectbox(
            "Select Cryptocurrency",
            options=list(analyzer.crypto_symbols.keys()),
            index=0
        )
        # Get the corresponding symbol
        symbol = analyzer.crypto_symbols[selected_crypto]
    with col2:
        timeframe = st.selectbox(
            "Select Timeframe:",
            ['1mo', '3mo', '6mo', '1y', '2y'],
            index=2
        )
    
    if st.button("Analyze"):
        with st.spinner('Analyzing cryptocurrency data...'):
            df, analysis, fig = analyzer.get_analysis(symbol, timeframe)
            
            if df is not None:
                # Display plot
                st.plotly_chart(fig, use_container_width=True)
                
                # Display analysis
                st.subheader("📊 Analysis")
                st.markdown(analysis)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${df['Close'][-1]:.2f}", 
                             f"{(df['Close'][-1] - df['Close'][-2]):.2f}")
                with col2:
                    st.metric("RSI", f"{df['RSI'][-1]:.2f}")
                with col3:
                    st.metric("MACD", f"{df['MACD'][-1]:.3f}")
                with col4:
                    st.metric("24h Volume", f"{df['Volume'][-1]:,.0f}")
            else:
                st.error(analysis)

if __name__ == "__main__":
    main()