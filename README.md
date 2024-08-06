# AI-Powered Threat Detection for Fintech

This project is a Streamlit application designed to detect threats in fintech environments using various machine learning models. The application allows users to upload a CSV file, preprocess the data, and make predictions using pre-trained models. The results are displayed with visualizations and suggestions for mitigation actions based on the threat level.

## Features

- Upload CSV files for prediction
- Preprocess data (remove duplicates, handle missing values, label encoding, scaling)
- Load and use multiple machine learning models
- Display model performance with conditional formatting
- Identify the best model and provide mitigation suggestions
- Visualize model performance and feature importance

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place your pre-trained model files in the project directory. The model file names should match those specified in the `model_files` dictionary in the code.

2. Run the Streamlit application:

    ```sh
    streamlit run app.py
    ```

3. Open your web browser and go to `http://localhost:8501` to access the application.

4. Upload your test CSV file and view the predictions and visualizations.

## File Structure

- `app.py`: Main application code
- `requirements.txt`: List of required Python packages
- `README.md`: Project documentation

## Dependencies

- streamlit
- pandas
- joblib
- scikit-learn
- plotly

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Joblib](https://joblib.readthedocs.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Plotly](https://plotly.com/)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## Contact

For any questions or inquiries, please contact [your-email@example.com].
