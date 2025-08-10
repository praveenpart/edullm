import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import re

# Page configuration
st.set_page_config(
    page_title="Academic Assistant Pro",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, modern UI
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Fix visibility issues */
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    
    .subject-card {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: white;
    }
    
    .answer-container {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        color: white;
    }
    
    .step-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
        color: white;
    }
    
    .formula-box {
        background: rgba(255, 193, 7, 0.1);
        border: 1px solid #ffc107;
        padding: 10px;
        border-radius: 4px;
        text-align: center;
        font-family: 'Courier New', monospace;
        margin: 10px 0;
        color: #ffc107;
    }
    
    .stSelectbox > div > div {
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    
    .stTextArea textarea {
        background-color: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    
    /* Fix text visibility */
    .stSelectbox label, .stTextArea label {
        color: white !important;
    }
    
    .stMarkdown, .stText {
        color: white;
    }
    
    /* Fix dropdown text */
    .stSelectbox > div > div > div {
        color: white;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced subject configurations with refined prompts
SUBJECTS = {
    "Mathematics": {
        "icon": "📐",
        "prompt": """You are an expert mathematics tutor with PhD-level knowledge. Provide step-by-step solutions that are:
- Mathematically rigorous and textbook-accurate
- Clearly explained with proper mathematical notation
- Include all intermediate steps
- Show work verification
- Use proper mathematical terminology
- Format equations clearly using standard notation""",
        "example": "Solve: 3x² - 12x + 9 = 0"
    },
    "Physics": {
        "icon": "⚡",
        "prompt": """You are a physics professor with expertise in all physics domains. Provide solutions that:
- Use correct physics principles and formulas
- Show dimensional analysis where applicable
- Include proper units throughout calculations
- Explain the physics concepts involved
- Draw connections to real-world applications
- Follow standard physics problem-solving methodology""",
        "example": "A 2kg object falls from 10m height. Find velocity just before impact."
    },
    "Chemistry": {
        "icon": "🧪",
        "prompt": """You are a chemistry expert with advanced knowledge. Provide solutions that:
- Use accurate chemical formulas and equations
- Balance chemical equations properly
- Include proper chemical nomenclature
- Show stoichiometric calculations step-by-step
- Explain molecular behavior and mechanisms
- Use standard chemistry notation and units""",
        "example": "Balance: Al + O₂ → Al₂O₃"
    },
    "Biology": {
        "icon": "🧬",
        "prompt": """You are a biology expert with comprehensive knowledge. Provide explanations that:
- Use accurate biological terminology
- Explain biological processes clearly
- Include relevant examples and analogies
- Connect concepts to real biological systems
- Use proper scientific classification
- Reference current biological understanding""",
        "example": "Explain the process of cellular respiration in detail."
    },
    "English Literature": {
        "icon": "📚",
        "prompt": """You are an English literature scholar with deep analytical skills. Provide analysis that:
- Uses proper literary terminology and concepts
- Includes textual evidence and citations
- Analyzes themes, symbols, and literary devices
- Considers historical and cultural context
- Follows academic writing standards
- Provides insightful interpretations""",
        "example": "Analyze the symbolism of light and darkness in Romeo and Juliet."
    },
    "History": {
        "icon": "🏛️",
        "prompt": """You are a history professor with expertise across time periods. Provide analysis that:
- Uses accurate historical facts and dates
- Considers multiple perspectives and sources
- Analyzes cause-and-effect relationships
- Includes relevant historical context
- Uses proper historical methodology
- Maintains objectivity while explaining significance""",
        "example": "Analyze the causes of World War I."
    },
    "Economics": {
        "icon": "💰",
        "prompt": """You are an economics expert with theoretical and practical knowledge. Provide explanations that:
- Use correct economic terminology and principles
- Include relevant graphs and models where applicable
- Explain both micro and macroeconomic concepts
- Use real-world examples and applications
- Show mathematical calculations for economic problems
- Connect theory to current economic conditions""",
        "example": "Explain supply and demand equilibrium with a market example."
    },
    "Computer Science": {
        "icon": "💻",
        "prompt": """You are a computer science expert with programming and theoretical knowledge. Provide solutions that:
- Use correct programming syntax and best practices
- Explain algorithms clearly with complexity analysis
- Include working code examples when relevant
- Explain computer science concepts thoroughly
- Use proper technical terminology
- Consider efficiency and optimization""",
        "example": "Implement binary search algorithm in Python."
    }
}

def should_show_diagram(question, subject):
    """
    Improved diagram detection - more permissive and better logic
    """
    question_lower = question.lower()
    
    # Strong visual indicators - if any of these are present, show diagram
    strong_visual_keywords = [
        'draw', 'sketch', 'plot', 'graph', 'construct', 'visualize', 
        'diagram', 'figure', 'chart', 'show graphically', 'illustrate',
        'represent visually', 'create diagram', 'make chart', 'display'
    ]
    
    # Check for strong visual indicators first
    for keyword in strong_visual_keywords:
        if keyword in question_lower:
            return True
    
    # Subject-specific visual keywords
    subject_keywords = {
        'Mathematics': [
            'parabola', 'quadratic', 'function', 'linear', 'curve', 'slope',
            'triangle', 'circle', 'rectangle', 'angle', 'coordinate',
            'sin', 'cos', 'tan', 'trigonometric', 'x²', 'x^2', 'y='
        ],
        'Physics': [
            'wave', 'frequency', 'amplitude', 'projectile', 'trajectory', 
            'motion', 'force diagram', 'circuit', 'field', 'vector'
        ],
        'Chemistry': [
            'reaction rate', 'concentration', 'molecular structure', 
            'lewis structure', 'phase diagram', 'orbital'
        ],
        'Biology': [
            'population growth', 'cell cycle', 'dna structure', 'ecosystem',
            'life cycle', 'growth curve', 'distribution'
        ],
        'Economics': [
            'supply', 'demand', 'curve', 'equilibrium', 'market',
            'production possibilities', 'cost curve'
        ]
    }
    
    # Check subject-specific keywords
    if subject in subject_keywords:
        for keyword in subject_keywords[subject]:
            if keyword in question_lower:
                return True
    
    # Pattern matching for visual requests
    visual_patterns = [
        r'graph.*function',
        r'plot.*points?',
        r'show.*relationship',
        r'y\s*=\s*.*x',  # equations like y = 2x + 3
        r'f\(x\)\s*=',   # function notation
    ]
    
    for pattern in visual_patterns:
        if re.search(pattern, question_lower):
            return True
    
    return False

def create_smart_visualization(question, subject):
    """Create intelligent visualizations based on question context and subject"""
    question_lower = question.lower()
    
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Mathematics visualizations
        if subject == "Mathematics":
            if any(term in question_lower for term in ['x²', 'x^2', 'quadratic', 'parabola']):
                x = np.linspace(-10, 10, 400)
                y = x**2
                ax.plot(x, y, 'b-', linewidth=2, label='y = x²')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.axvline(x=0, color='k', linewidth=0.5)
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.set_title('Quadratic Function', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                
            elif any(term in question_lower for term in ['linear', 'y=', 'slope', 'line', 'function']) and 'x' in question_lower:
                x = np.linspace(-10, 10, 100)
                y = 2*x + 3
                ax.plot(x, y, 'r-', linewidth=2, label='y = 2x + 3')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.axvline(x=0, color='k', linewidth=0.5)
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.set_title('Linear Function', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                
            elif any(term in question_lower for term in ['sin', 'cos', 'tan', 'trigonometric']):
                x = np.linspace(-2*np.pi, 2*np.pi, 400)
                y1 = np.sin(x)
                y2 = np.cos(x)
                ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
                ax.plot(x, y2, 'r-', linewidth=2, label='cos(x)')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.axvline(x=0, color='k', linewidth=0.5)
                ax.set_xlabel('x (radians)', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.set_title('Trigonometric Functions', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                
            elif any(term in question_lower for term in ['circle', 'radius', 'circumference']):
                theta = np.linspace(0, 2*np.pi, 100)
                r = 5
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ax.plot(x, y, 'b-', linewidth=2, label=f'Circle (r={r})')
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.axvline(x=0, color='k', linewidth=0.5)
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.set_title('Circle', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                
            else:
                # Default math visualization - coordinate plane
                x = np.linspace(-5, 5, 100)
                y = x**2 - 4
                ax.plot(x, y, 'b-', linewidth=2, label='Example: y = x² - 4')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linewidth=0.5)
                ax.axvline(x=0, color='k', linewidth=0.5)
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.set_title('Mathematical Function', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
        
        # Physics visualizations
        elif subject == "Physics":
            if any(term in question_lower for term in ['wave', 'frequency', 'amplitude']):
                t = np.linspace(0, 4*np.pi, 400)
                y = np.sin(t)
                ax.plot(t, y, 'b-', linewidth=2, label='Wave Function')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time/Position', fontsize=12)
                ax.set_ylabel('Amplitude', fontsize=12)
                ax.set_title('Wave Pattern', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                
            elif any(term in question_lower for term in ['projectile', 'trajectory', 'motion']):
                t = np.linspace(0, 5, 100)
                x = 10 * t  # horizontal motion
                y = 10 * t - 0.5 * 9.8 * t**2  # vertical motion with gravity
                y[y < 0] = 0  # ground level
                ax.plot(x, y, 'r-', linewidth=2, label='Projectile Path')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Horizontal Distance (m)', fontsize=12)
                ax.set_ylabel('Height (m)', fontsize=12)
                ax.set_title('Projectile Motion', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
            else:
                # Default physics diagram
                t = np.linspace(0, 2*np.pi, 100)
                y = np.sin(t)
                ax.plot(t, y, 'g-', linewidth=2, label='Physics Function')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Parameter', fontsize=12)
                ax.set_ylabel('Value', fontsize=12)
                ax.set_title('Physics Visualization', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
        
        # Chemistry visualizations
        elif subject == "Chemistry":
            if any(term in question_lower for term in ['reaction rate', 'concentration', 'time']):
                t = np.linspace(0, 10, 100)
                concentration = 1.0 * np.exp(-0.3 * t)  # exponential decay
                ax.plot(t, concentration, 'g-', linewidth=2, label='Concentration vs Time')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel('Concentration (M)', fontsize=12)
                ax.set_title('Reaction Kinetics', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
            else:
                # Default chemistry diagram
                t = np.linspace(0, 10, 100)
                conc = np.exp(-0.2 * t)
                ax.plot(t, conc, 'orange', linewidth=2, label='Chemical Process')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Concentration', fontsize=12)
                ax.set_title('Chemical Process', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
        
        # Biology visualizations
        elif subject == "Biology":
            if any(term in question_lower for term in ['population', 'growth', 'exponential']):
                t = np.linspace(0, 10, 100)
                population = 100 * np.exp(0.2 * t)
                ax.plot(t, population, 'g-', linewidth=2, label='Population Growth')
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Population Size', fontsize=12)
                ax.set_title('Exponential Population Growth', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
            elif any(term in question_lower for term in ['cell cycle', 'mitosis', 'phases']):
                phases = ['G1', 'S', 'G2', 'M']
                sizes = [25, 30, 20, 25]
                colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
                ax.pie(sizes, labels=phases, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('Cell Cycle Phases', fontsize=14, fontweight='bold')
            else:
                # Default biology diagram
                t = np.linspace(0, 20, 100)
                growth = 50 * (1 - np.exp(-0.3 * t))
                ax.plot(t, growth, 'green', linewidth=2, label='Biological Growth')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Time', fontsize=12)
                ax.set_ylabel('Growth', fontsize=12)
                ax.set_title('Biological Process', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
        
        # Economics visualizations
        elif subject == "Economics":
            if any(term in question_lower for term in ['supply', 'demand', 'curve', 'equilibrium']):
                quantity = np.linspace(0, 10, 100)
                supply = 2 * quantity  # supply curve
                demand = 20 - quantity  # demand curve
                ax.plot(quantity, supply, 'b-', linewidth=2, label='Supply')
                ax.plot(quantity, demand, 'r-', linewidth=2, label='Demand')
                
                # Find equilibrium point
                eq_quantity = 20/3
                eq_price = 2 * eq_quantity
                ax.plot(eq_quantity, eq_price, 'go', markersize=8, label='Equilibrium')
                
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Quantity', fontsize=12)
                ax.set_ylabel('Price', fontsize=12)
                ax.set_title('Supply and Demand', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
            else:
                # Default economics diagram
                x = np.linspace(0, 10, 100)
                supply = x * 2
                demand = 20 - x
                ax.plot(x, supply, 'blue', linewidth=2, label='Economic Function 1')
                ax.plot(x, demand, 'red', linewidth=2, label='Economic Function 2')
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Quantity', fontsize=12)
                ax.set_ylabel('Price', fontsize=12)
                ax.set_title('Economic Analysis', fontsize=14, fontweight='bold')
                ax.legend(fontsize=11)
        
        # For other subjects, create a generic visualization
        else:
            x = np.linspace(-5, 5, 100)
            y = np.sin(x)
            ax.plot(x, y, 'purple', linewidth=2, label='General Function')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('X', fontsize=12)
            ax.set_ylabel('Y', fontsize=12)
            ax.set_title(f'{subject} Visualization', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
        
        # Save plot to bytes
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=150, facecolor='white')
        buf.seek(0)
        plt.close(fig)
        
        return buf
        
    except Exception as e:
        plt.close('all')
        return None

def get_api_response(question, subject):
    """Get response from OpenRouter API with enhanced prompting"""
    
    # Check if API key exists
    if 'OPENROUTER_API_KEY' not in st.secrets:
        st.error("⚠️ API key not configured. Please add OPENROUTER_API_KEY to Streamlit secrets.")
        return None
    
    api_key = st.secrets['OPENROUTER_API_KEY']
    
    # Enhanced system prompt - more detailed for better answers
    system_prompt = f"""
    You are an expert {subject} tutor providing comprehensive, step-by-step solutions.
    
    CRITICAL REQUIREMENTS:
    1. Provide DETAILED explanations with multiple steps
    2. Show ALL mathematical work and intermediate calculations
    3. Use clear mathematical notation (write fractions as 3/2, exponents as x^2, square roots as sqrt())
    4. Explain WHY each step is taken
    5. Include verification/checking of the final answer
    6. For math problems: show the formula first, then substitute values, then calculate
    7. Break complex problems into smaller, manageable steps
    8. Be thorough and educational - students should understand the process completely
    
    FORMATTING:
    - Use "Step 1:", "Step 2:", etc. for clear organization
    - Put final answers in a separate "Final Answer:" section
    - Use simple text formatting (no LaTeX symbols like \\frac or \\sqrt)
    - Write equations clearly: x = (-b ± sqrt(b^2 - 4ac)) / (2a)
    
    Subject: {subject}
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "temperature": 0.1,
        "max_tokens": 1200
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network Error: {str(e)}")
        return None

def format_math_response(response_text):
    """Format mathematical responses with proper styling and convert LaTeX frac/sqrt to plain text."""
    
    formatted = response_text
    
    # Convert nested fractions recursively - multiple passes to catch all
    def replace_frac(match):
        numerator = match.group(1)
        denominator = match.group(2)
        # Recursively convert numerator and denominator (to handle nested fracs)
        numerator_conv = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', replace_frac, numerator)
        denominator_conv = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', replace_frac, denominator)
        return f"({numerator_conv})/({denominator_conv})"
    
    # Multiple passes to ensure all fractions are converted
    for _ in range(5):  # Up to 5 levels of nesting
        if r'\frac{' not in formatted:
            break
        formatted = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', replace_frac, formatted)
    
    # Handle any remaining simple fractions with a basic pattern
    formatted = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', formatted)
    
    # Replace \sqrt{...} with sqrt(...)
    formatted = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', formatted)
    
    # Remove various LaTeX delimiters
    formatted = re.sub(r'\\\(|\\\)', '', formatted)
    formatted = re.sub(r'\\\[|\\\]', '', formatted)
    formatted = re.sub(r'\\left\(', '(', formatted)
    formatted = re.sub(r'\\right\)', ')', formatted)
    
    # Replace common LaTeX symbols
    formatted = re.sub(r'\\cdot', '*', formatted)
    formatted = re.sub(r'\\pm', '±', formatted)
    formatted = re.sub(r'\\times', '×', formatted)
    formatted = re.sub(r'\\div', '÷', formatted)
    
    # Clean up any remaining backslashes followed by letters (LaTeX commands)
    formatted = re.sub(r'\\[a-zA-Z]+', '', formatted)
    
    # Format for HTML display
    lines = formatted.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append('<br>')
            continue
        
        # Format step headers (including **Step patterns)
        if re.match(r'^\*\*Step \d+:', line) or re.match(r'^Step \d+:', line):
            # Remove markdown bold formatting
            clean_line = re.sub(r'\*\*', '', line)
            processed_lines.append(f'<div class="step-box"><strong>{clean_line}</strong></div>')
        
        # Format section headers
        elif line.startswith('**') and line.endswith('**'):
            clean_line = line.replace('**', '')
            processed_lines.append(f'<div class="step-box"><strong>{clean_line}</strong></div>')
        
        # Format equations (lines with = and mathematical content)
        elif '=' in line and any(ch in line for ch in ['x', '+', '-', '*', '/', '^', 'sqrt', '(', ')']):
            processed_lines.append(f'<div class="math-step">{line}</div>')
        
        # Format final answers
        elif (line.startswith('Final Answer:') or line.startswith('Therefore') or 
              line.startswith('Answer:') or 'solutions to the equation' in line):
            processed_lines.append(f'<div class="formula-box">{line}</div>')
        
        # Format solution headers
        elif line.startswith('Solution:') or line.startswith('Given:') or line.startswith('To solve'):
            processed_lines.append(f'<div class="step-box"><strong>{line}</strong></div>')
        
        # Regular text
        else:
            processed_lines.append(f'<p style="margin: 8px 0; color: white;">{line}</p>')
    
    return ''.join(processed_lines)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎓 Academic Assistant Pro</h1>
        <p>Expert-level homework assistance across all subjects</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📖 Select Subject")
        
        # Subject selection with icons
        subject_options = [f"{info['icon']} {subject}" for subject, info in SUBJECTS.items()]
        selected_subject_display = st.selectbox(
            "Choose your subject:",
            subject_options,
            help="Select the academic subject for your question"
        )
        
        # Extract actual subject name
        selected_subject = selected_subject_display.split(' ', 1)[1]
        
        # Show example question
        st.markdown("### 💡 Example Question")
        with st.expander("See example"):
            st.info(f"**{selected_subject}**: {SUBJECTS[selected_subject]['example']}")
    
    with col2:
        st.markdown("### ❓ Your Question")
        
        # Question input
        question = st.text_area(
            "Enter your homework question:",
            height=120,
            placeholder=f"Ask your {selected_subject} question here...",
            help="Be specific and include all relevant details"
        )
        
        # Solve button
        if st.button("🎯 Get Solution", type="primary"):
            if question.strip():
                with st.spinner(f"Analyzing your {selected_subject} question..."):
                    # Get AI response
                    response = get_api_response(question, selected_subject)
                    
                    if response:
                        # Display clean answer
                        st.markdown("### 📝 Solution")
                        
                        # Format response based on subject
                        if selected_subject == "Mathematics":
                            formatted_response = format_math_response(response)
                        else:
                            formatted_response = response.replace('\n\n', '<br><br>')
                        
                        with st.container():
                            st.markdown(f"""
                            <div class="answer-container">
                                <h4>{SUBJECTS[selected_subject]['icon']} {selected_subject} Solution</h4>
                                <div>
                                    {formatted_response}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Smart diagram detection and generation
                        st.markdown("### 🔍 Diagram Detection Debug")
                        diagram_should_show = should_show_diagram(question, selected_subject)
                        st.info(f"Should show diagram: **{diagram_should_show}**")
                        st.info(f"Question: '{question}'")
                        st.info(f"Subject: '{selected_subject}'")
                        
                        if diagram_should_show:
                            with st.spinner("Creating visualization..."):
                                viz = create_smart_visualization(question, selected_subject)
                                if viz:
                                    st.markdown("### 📊 Visual Representation")
                                    st.image(viz, use_container_width=True, caption=f"{selected_subject} Visualization")
                                    st.success("✅ Diagram generated successfully!")
                                else:
                                    st.warning("⚠️ Could not generate visualization for this question.")
                        else:
                            st.info("ℹ️ No diagram needed for this question type.")
                        
                        # Feedback section
                        st.markdown("### 📊 Rate this solution")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            if st.button("👍 Helpful"):
                                st.success("Thank you for your feedback!")
                        with col_b:
                            if st.button("👎 Needs improvement"):
                                st.info("We'll work on improving our responses!")
                        with col_c:
                            if st.button("🔄 Try again"):
                                st.rerun()
                    
            else:
                st.warning("⚠️ Please enter a question to get help.")
    
    # Test cases section
    st.markdown("---")
    st.markdown("### 🧪 Test Diagram Detection")
    st.markdown("Try these test cases to see diagram detection in action:")
    
    test_cases = {
        "Mathematics": [
            "Draw a parabola for y = x²",
            "Graph the function y = 2x + 3",
            "Sketch a circle with radius 5",
            "Plot the sine function",
            "Solve: 2x + 3 = 7"  # Should NOT show diagram
        ],
        "Physics": [
            "Draw a wave with frequency 5Hz",
            "Show projectile motion trajectory",
            "Calculate the momentum"  # Should NOT show diagram
        ],
        "Economics": [
            "Draw supply and demand curves",
            "Show market equilibrium",
            "What is inflation?"  # Should NOT show diagram
        ]
    }
    
    col_test1, col_test2, col_test3 = st.columns(3)
    
    with col_test1:
        st.markdown("**Mathematics Tests:**")
        for test in test_cases["Mathematics"]:
            if st.button(f"Test: {test[:20]}...", key=f"math_{test}"):
                st.text_area("Auto-filled question:", value=test, key=f"result_{test}")
                should_show = should_show_diagram(test, "Mathematics")
                if should_show:
                    st.success(f"✅ Would show diagram")
                else:
                    st.info(f"ℹ️ Would NOT show diagram")
    
    with col_test2:
        st.markdown("**Physics Tests:**")
        for test in test_cases["Physics"]:
            if st.button(f"Test: {test[:20]}...", key=f"phys_{test}"):
                st.text_area("Auto-filled question:", value=test, key=f"result_{test}")
                should_show = should_show_diagram(test, "Physics")
                if should_show:
                    st.success(f"✅ Would show diagram")
                else:
                    st.info(f"ℹ️ Would NOT show diagram")
    
    with col_test3:
        st.markdown("**Economics Tests:**")
        for test in test_cases["Economics"]:
            if st.button(f"Test: {test[:20]}...", key=f"econ_{test}"):
                st.text_area("Auto-filled question:", value=test, key=f"result_{test}")
                should_show = should_show_diagram(test, "Economics")
                if should_show:
                    st.success(f"✅ Would show diagram")
                else:
                    st.info(f"ℹ️ Would NOT show diagram")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎓 Academic Assistant Pro - Powered by Advanced AI</p>
        <p><small>Providing accurate, textbook-quality educational assistance with smart diagram detection</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()