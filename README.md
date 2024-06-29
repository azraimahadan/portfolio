# Portfolio of Personal Project

## Llama3 QnA Chatbot

This project involves the creation of an advanced chatbot utilizing the **LlamaIndex** framework. The chatbot is hosted on **Kaggle** via tunnel. Access the notebook [here](https://www.kaggle.com/code/azraimohamad/llamaindex-malaysia-qa-bot) to run the code by yourself(Copy and Edit the notebook). The core of the chatbot is powered by the latest **Llama3 model** by Meta, known for its robust performance and accuracy.

### Key Features

- **Llama3 Model**: Leveraging Meta's state-of-the-art Llama3 model, which provides advanced natural language processing capabilities.
- **Hosting on Kaggle**: The chatbot is hosted on Kaggle, utilizing free GPU for resource-constrainted developers.
- **Advanced Retrieval Techniques**: Implementing **Retrieval-Augmented Generation (RAG)** with a **recursive retriever** to enhance the accuracy of fetched facts and responses.
  
### Technical Details

1. **LlamaIndex Framework**:
   - Utilized for its flexibility and comprehensive features in chatbot development.
   - Supports integration with the latest NLP models and retrieval techniques.

2. **Hosting on Kaggle**:
   - Kaggle provides a robust platform for hosting and interacting with the chatbot.
   - Tunnel setup allows for easy and secure access.

3. **RAG with Recursive Retriever**:
   - **Retrieval-Augmented Generation (RAG)**: A technique that enhances the response generation by retrieving relevant documents or facts before generating an answer.
   - **Recursive Retriever**: Ensures the accuracy of the retrieved information by recursively refining the retrieval process.

### Future Enhancements

- Integration with more advanced models and techniques as they become available.
- Continuous improvement of the retrieval and generation process to ensure high-quality responses.
- Expansion of the chatbot's capabilities to cover more use cases and domains.

- [Kaggle Notebook](https://www.kaggle.com/code/azraimohamad/llamaindex-malaysia-qa-bot)
  
## Real-Time Face Analysis for Customer Segmentation

Build Flask web application that utilizes a convolutional deep neural network model to detect faces in real-time webcam feeds from scratch. Upon detection, the application predicts the age, gender, and ethnicity of the individuals, displaying the results alongside the live video stream. The application continuously captures frames, analyzes them, and provides immediate feedback on the identified attributes of the detected faces. The results are automatically sent and analyzed to provide demographical analysis in Excel.

![alt text](./images/a1.PNG)

Users can interact with the application by adjusting verification criteria for gender, race, and age, receiving immediate feedback on the accuracy of predictions.

![alt text](./images/a2.PNG)


- [Github Repo](https://github.com/azraimahadan/Face-Analysis-for-Customer-Segmentation)
- [Brochure](https://github.com/azraimahadan/Face-Analysis-for-Customer-Segmentation/blob/main/poster.PNG)

Technologies used:  
- Model Development - Python, Tensorflow, OpenCV, Pandas
- Model Deployment & Monitoring - Flask, Excel

## Skills Extracting with BERT

Build an application leveraging BERT language model for skill extraction from job descriptions. This tool employs advanced natural language processing techniques and fine-tuned LLM to automatically identify and extract essential skills, enhancing the efficiency of talent acquisition processes

<!-- ![alt text](./images/b1.PNG) -->
<img src="./images/b1.PNG" width="350" height="300">

- [Github Repo](https://github.com/azraimahadan/skill-extraction-with-bert)
- [Streamlit App](https://skill-extraction-with-bert-yvf7zfcahggh5zyi6zfgwn.streamlit.app/)
- [HuggingFace space](https://huggingface.co/spaces/azrai99/Skills-Extraction-from-Job-Post)


Technologies used: Python, Pytorch, Streamlit, HuggingFace

## Prophet Forecasting App

Build an application for time series forecasting using Facebook Prophet. This application allows users to upload CSV files containing time series data, preprocesses the data, and generates forecasts for future time periods. It utilizes the Prophet library to fit models and visualize forecast components, enabling users to gain insights into the underlying trends and patterns in their data

![alt text](./images/c1.PNG)

![alt text](./images/c2.PNG)

- [Github Repo](https://github.com/azraimahadan/prophet-forecast)
- [Web App](https://prophet-forecast-jyorxovb3vkvsv7ewbwmkq.streamlit.app/)

Technologies used: Python, Fbprophet, Pandas Streamlit

## Linking Writing Style with Essay Score 

Build 2 types of ML model, Light Gradient Boost Machine(LGBM) and Linear Regression to explore the relationship between learners’ writing behaviors and writing performance, which could provide valuable insights for intelligent tutoring systems. The project leverage Optuna as a faster way to optimizing hyperarameter of LGBM to provide better model performance.

![alt text](./images/d1.png)
- [Github](https://github.com/azraimahadan/portfolio/tree/main/Linking%20writing%20with%20Essay%20Score)

Technologies used: Python, LGBM, Linear Regression, sklearn, Optuna, matplotlib
