# Smart Recycling Assistant

A web application designed to help users identify objects in images and provide recommendations on how to dispose of them sustainably.

## Overview

This project uses TensorFlow for object detection and OpenAI for generating insights about the detected objects. The application is built with Streamlit, providing a user-friendly interface for uploading images and viewing disposal recommendations.

## Features

- **Object Detection**: Uses TensorFlow's InceptionV3 model to identify objects in uploaded images.
- **Insight Generation**: Leverages OpenAI's GPT model to generate insights about the detected objects, including their materials and recommended disposal methods.
- **Disposal Recommendations**: Provides users with actionable advice on how to dispose of detected objects sustainably.

## Requirements

- **Python 3.8+**: Required for running the application.
- **TensorFlow**: For object detection.
- **OpenAI API Key**: For generating insights.
- **Streamlit**: For the web interface.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SuhasPalani/smart-recycling-assistant.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install tensorflow pillow streamlit openai python-dotenv
   ```

3. **Set Up API Keys**:
   Create a `.env` file in the project root with your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your-openai-api-key-here
   ```

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload an Image**:
   Use the file uploader to select an image of an object you want to analyze.

2. **View Results**:
   The application will display the detected object, generated insight, and recommended disposal method.

## Contributing

Contributions are welcome! If you have ideas for improving the application or want to report issues, please open a pull request or issue on GitHub.

