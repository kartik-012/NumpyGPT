# NumPyGPT  🐍🤖

Your intelligent partner for mastering NumPy! This AI-powered application, built with the Gemini API, helps you generate, understand, and debug NumPy code through a simple, conversational interface.


<!-- You can add a screenshot or GIF of your application here -->
<!-- ![App Screenshot](link-to-your-screenshot.png) -->

## ✨ Features

*   **Natural Language to NumPy:** Describe what you want to do in plain English, and get the corresponding NumPy code.
*   **Code Explanation:** Paste a NumPy snippet and receive a detailed explanation of what it does.
*   **Interactive Chat:** A user-friendly interface for interacting with the AI.
*   **Powered by Gemini:** Leverages Google's powerful Gemini model for accurate and context-aware responses.

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   [Node.js](https://nodejs.org/) (v18 or later recommended)
*   [npm](https://www.npmjs.com/) (comes with Node.js)
*   A [Gemini API Key](https://makersuite.google.com/app/apikey).

### 💻 Installation & Running Locally

1.  **Clone the repository (or download the source):**
    ```bash
    git clone <your-repository-url>
    cd <repository-folder>
    ```

2.  **Install dependencies:**
    ```bash
    npm install
    ```

3.  **Set up your environment variables:**
    Create a file named `.env.local` in the root of your project and add your Gemini API key:
    ```
    GEMINI_API_KEY="YOUR_API_KEY_HERE"
    ```
    **Important:** The `.gitignore` file is configured to prevent this file from being committed. Never share your API key publicly.

4.  **Run the development servers:**
    This project has a separate frontend and backend. You'll need to run them in two separate terminal windows.

    *Terminal 1: Start the Backend*
    ```bash
    npm run dev:backend
    ```

    *Terminal 2: Start the Frontend*
    ```bash
    npm run dev:frontend
    ```


Open the local URL provided by the frontend server (usually http://localhost:5173) in your browser to see the result.

## 🛠️ Built With

*   **Frontend:** React + Vite
*   **Backend:** Express.js
*   **AI:** Google Gemini API
*   **Runtime:** Node.js 
