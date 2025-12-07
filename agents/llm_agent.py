"""
Module: llm_agent.py
Description: Real LLM agent with context-aware preprocessing suggestions
Author: Raj Nandini
Date: 2025-10-28
"""

import os
import json
from typing import Optional, Dict, Any
import pandas as pd


class LLMAgent:
    """Real LLM agent for preprocessing suggestions."""
    
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None, provider: str = "openai"):
        """
        Initialize LLM Agent.
        
        Args:
            model: Model name (gpt-3.5-turbo, gpt-4, claude-3-sonnet, mixtral-8x7b, etc.)
            api_key: API key (or read from env)
            provider: "openai", "anthropic", or "groq"
        """
        self.model = model
        self.provider = provider.lower()
        
        # Auto-detect API key from environment
        env_key = f"{provider.upper()}_API_KEY"
        self.api_key = api_key or os.getenv(env_key)
        self.enabled = bool(self.api_key)
        
        if not self.enabled:
            print(f"⚠️ {provider.upper()} API key not found in {env_key}")
            print(f"   Set: export {env_key}='your-key-here'")
        else:
            print(f"✅ {provider.upper()} LLM enabled ({model})")
    
    def _build_context_prompt(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Build rich context-aware prompt."""
        if not context:
            return question
        
        context_str = "\n\n### DATA CONTEXT ###\n"
        
        if isinstance(context.get("df"), pd.DataFrame):
            df = context["df"]
            context_str += f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
            context_str += f"Column Types: {df.dtypes.value_counts().to_dict()}\n"
            context_str += f"Missing Values: {df.isnull().sum().sum()} total ({df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 100:.1f}%)\n"
            
            # Numeric columns stats
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                context_str += f"Numeric Columns: {len(numeric_cols)} columns\n"
                context_str += f"Skewness Range: {df[numeric_cols].skew().min():.2f} to {df[numeric_cols].skew().max():.2f}\n"
            
            # Categorical columns stats
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                context_str += f"Categorical Columns: {len(cat_cols)} columns\n"
                cardinality = [df[col].nunique() for col in cat_cols]
                context_str += f"Cardinality Range: {min(cardinality)} to {max(cardinality)}\n"
            
            # Target imbalance (OPTIONAL - only if target_col exists and is in df)
            target_col = context.get("target_col")
            if target_col and target_col in df.columns:
                target_dist = df[target_col].value_counts(normalize=True)
                context_str += f"Target Distribution: {target_dist.to_dict()}\n"
        
        if context.get("current_stage"):
            context_str += f"Current Stage: {context['current_stage']}\n"
        
        if context.get("issues"):
            context_str += f"Identified Issues: {', '.join(context['issues'])}\n"
        
        return f"{question}\n{context_str}"
    
    def ask(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Ask LLM for suggestions.
        
        Args:
            question: The question to ask
            context: Optional context dict with:
                - df: DataFrame
                - target_col: Target column name (OPTIONAL)
                - current_stage: Current pipeline stage
                - issues: List of identified issues
        
        Returns:
            LLM response or mock response if disabled
        """
        if not self.enabled:
            return self._mock_response(question, context)
        
        try:
            # Build rich context prompt
            rich_prompt = self._build_context_prompt(question, context)
            
            if self.provider == "openai":
                return self._ask_openai(rich_prompt, context)
            elif self.provider == "anthropic":
                return self._ask_claude(rich_prompt, context)
            elif self.provider == "groq":
                return self._ask_groq(rich_prompt, context)
            else:
                print(f"⚠️ Unknown provider: {self.provider}")
                return self._mock_response(question, context)
        except Exception as e:
            print(f"⚠️ LLM error ({self.provider}): {e}")
            return self._mock_response(question, context)
    
    def _ask_openai(self, question: str, context: Optional[Dict] = None) -> str:
        """Ask OpenAI API."""
        try:
            import openai
            openai.api_key = self.api_key
            
            system_prompt = """You are an expert data scientist specializing in data preprocessing.
Analyze the provided dataset context and question carefully.
Provide SPECIFIC, ACTIONABLE suggestions tailored to THIS dataset, not generic advice.
Include:
- Why this approach fits the data characteristics
- Implementation details or code hints
- Expected outcomes
Keep responses concise (2-3 sentences max)."""
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=200,
                timeout=10
            )
            
            return response.choices[0].message.content.strip()
        
        except ImportError:
            print("⚠️ Install: pip install openai")
            return self._mock_response(question, context)
        except Exception as e:
            print(f"⚠️ OpenAI error: {e}")
            return self._mock_response(question, context)
    
    def _ask_claude(self, question: str, context: Optional[Dict] = None) -> str:
        """Ask Claude API."""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model=self.model,
                max_tokens=200,
                system="""You are an expert data scientist specializing in data preprocessing.
Analyze the provided dataset context and question carefully.
Provide SPECIFIC, ACTIONABLE suggestions tailored to THIS dataset.
Include why this approach fits, implementation hints, and expected outcomes.
Keep responses concise but detailed.""",
                messages=[
                    {"role": "user", "content": question}
                ]
            )
            
            return response.content[0].text.strip()
        
        except ImportError:
            print("⚠️ Install: pip install anthropic")
            return self._mock_response(question, context)
        except Exception as e:
            print(f"⚠️ Claude error: {e}")
            return self._mock_response(question, context)
    
    def _ask_groq(self, question: str, context: Optional[Dict] = None) -> str:
        """Ask Groq API (fastest LLM inference)."""
        try:
            from groq import Groq
            
            client = Groq(api_key=self.api_key)
            
            system_prompt = """You are an expert data scientist specializing in data preprocessing.
Analyze the provided dataset context and question carefully.
Provide SPECIFIC, ACTIONABLE suggestions tailored to THIS dataset.
Include why this approach fits, implementation hints, and expected outcomes.
Keep responses concise but detailed."""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=200,
                timeout=10
            )
            
            return response.choices[0].message.content.strip()
        
        except ImportError:
            print("⚠️ Install: pip install groq")
            return self._mock_response(question, context)
        except Exception as e:
            print(f"⚠️ Groq error: {e}")
            return self._mock_response(question, context)
    
    def _mock_response(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate context-aware mock LLM response (when API disabled)."""
        # Extract context (all optional)
        df = context.get("df") if context else None
        stage = context.get("current_stage", "").lower() if context else ""
        issues = context.get("issues", []) if context else []
        
        # Stage-specific responses with data awareness
        if "ingestion" in stage or "load" in question.lower():
            if df is not None:
                missing_pct = df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 100
                return f"💡 Data loaded: {df.shape[0]} rows × {df.shape[1]} cols. Missing: {missing_pct:.1f}%. Analyze missing patterns; consider domain-specific vs statistical imputation."
            return "💡 Data loaded successfully. Analyze missing value patterns and distributions before proceeding."
        
        if "clean" in stage or "missing" in question.lower():
            if df is not None:
                missing_pct = df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 100
                if missing_pct > 30:
                    return f"💡 High missingness ({missing_pct:.1f}%): Use KNN imputation for continuous, mode for categorical. Consider creating 'missing' indicator features."
                elif missing_pct > 5:
                    return f"💡 Moderate missingness ({missing_pct:.1f}%): Use domain knowledge for imputation; avoid deletion. Test MCAR assumption."
                else:
                    return f"💡 Low missingness ({missing_pct:.1f}%): Use mean/median for numeric, mode for categorical. Simple deletion acceptable if <5%."
            return "💡 Implement missing value strategy based on percentage: <5% (delete), 5-30% (impute), >30% (indicator features)."
        
        if "transform" in stage or "scaling" in question.lower():
            if df is not None:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    skewness = df[numeric_cols].skew().abs().max()
                    if skewness > 2:
                        return f"💡 High skewness detected ({skewness:.2f}): Use RobustScaler or log transformation. Avoid StandardScaler (sensitive to outliers)."
                    elif skewness > 1:
                        return f"💡 Moderate skewness ({skewness:.2f}): RobustScaler recommended. Consider Box-Cox transformation for extreme cases."
                    else:
                        return f"💡 Low skewness ({skewness:.2f}): StandardScaler appropriate. MinMaxScaler if preserving sparse patterns important."
            return "💡 Choose scaler based on distribution: StandardScaler (normal), RobustScaler (skewed/outliers), MinMaxScaler (sparse data)."
        
        if "feature" in stage or "engineer" in question.lower():
            if df is not None:
                n_cols = df.shape[1]
                n_numeric = len(df.select_dtypes(include=['number']).columns)
                if n_numeric > 3:
                    return f"💡 Create interaction features (e.g., col1×col2) for top {min(3, n_numeric)} numeric columns. Use feature importance to validate."
                return f"💡 Limited numeric columns ({n_numeric}). Focus on domain-specific features and polynomial expansions with regularization."
            return "💡 Create interaction features, polynomial expansions, and domain-specific transformations. Validate with feature importance."
        
        if "eda" in stage or "analyze" in question.lower():
            if df is not None and len(issues) > 0:
                return f"💡 Identified issues: {', '.join(issues)}. Create targeted visualizations; compute correlation/VIF for multicollinearity check."
            return "💡 Analyze distributions, correlations, and patterns. Identify outliers, multicollinearity, and class imbalance."
        
        if "encode" in question.lower() or "categorical" in question.lower():
            if df is not None:
                cat_cols = df.select_dtypes(include=['object']).columns
                if len(cat_cols) > 0:
                    cardinality = [df[col].nunique() for col in cat_cols]
                    high_card = sum(1 for c in cardinality if c > 10)
                    if high_card > 0:
                        return f"💡 {high_card} high-cardinality columns detected. Use target encoding + regularization or frequency encoding; avoid one-hot (curse of dimensionality)."
                    else:
                        return f"💡 Low cardinality: One-hot encode for <10 categories. Use LabelEncoder for ordinal features."
            return "💡 One-hot for nominal features (<10 categories), target encoding for high-cardinality, LabelEncoder for ordinal."
        
        if "split" in stage or "train" in question.lower():
            return "💡 Use stratified train-test split (preserve class distribution). Consider temporal split if time-series data."
        
        if "imbalance" in question.lower() or "balance" in question.lower():
            return "💡 Use SMOTE for minority oversampling, or adjust class weights. Evaluate with F1/ROC-AUC, not accuracy."
        
        return "💡 Analyze data characteristics; apply transformations iteratively; validate assumptions; measure impact with holdout set."


def create_llm_agent(use_llm: bool = False, provider: str = "openai", model: str = "gpt-3.5-turbo") -> Optional[LLMAgent]:
    """
    Factory function to create LLM agent.
    
    Args:
        use_llm: Enable/disable LLM features
        provider: "openai", "anthropic", or "groq"
        model: Model name
    
    Returns:
        LLMAgent instance or None
    """
    if not use_llm:
        return None
    
    # Default models per provider
    model_defaults = {
        "openai": "gpt-3.5-turbo",
        "anthropic": "claude-3-sonnet-20240229",
        "groq": "llama-3.1-70b-versatile"
    }
    
    if model == "gpt-3.5-turbo" and provider != "openai":
        model = model_defaults.get(provider, model)
    
    agent = LLMAgent(model=model, provider=provider)
    return agent if agent.enabled else None