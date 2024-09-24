import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

file_path = 'Dataset_CyberCrime_Sean.csv'
data = pd.read_csv(file_path)

model_filename = "best_random_forest_model.pkl"
scaler_filename = "scaler.pkl"

def main():
    st.sidebar.title("Crime Rate Prediction")
    page = st.sidebar.radio("Go to", ["List of Cities", "Total Crime Rate"])

    if page == "List of Cities":
        list_of_cities_page()
    elif page == "Total Crime Rate":
        total_crime_rate_page()

def list_of_cities_page():
    st.title("List of Cities")
    city_list = data['City'].unique()
    city_df = pd.DataFrame(city_list, columns=['City'])
    city_df.index += 1  # Start index from 1
    city_df.index.name = 'Index'
    
    # Remove the first row and display the rest
    city_df = city_df.iloc[1:]
    
    # Display the table with default index width and city names in the second row
    st.write(city_df)

def total_crime_rate_page():
    st.title("Total Crime Rate")
    city_name = st.text_input("Enter the city name:")
    if st.button("Submit"):
        st.write("")  # Add extra space after the button
        if city_name:
            city_data = data[data['City'] == city_name]
            if city_data.empty:
                st.error(f"City '{city_name}' not found in the dataset.")
            else:
                try:
                    loaded_model = joblib.load(model_filename)
                    loaded_scaler = joblib.load(scaler_filename)

                    city_features = city_data.drop(['City', 'Total'], axis=1)
                    scaled_city_features = loaded_scaler.transform(city_features)
                    predicted_total_crimes = loaded_model.predict(scaled_city_features)[0]

                    total_category_crimes = city_features.iloc[0]
                    category_percentages = (total_category_crimes / total_category_crimes.sum()) * 100
                    sorted_categories = category_percentages.sort_values(ascending=False)

                    st.subheader(f"Predicted Total Crimes for {city_name}: {int(predicted_total_crimes)}")
                    st.subheader("Percentage Breakdown of Crimes by Category:")
                    
                    for category, percentage in sorted_categories.items():
                        if percentage > 0 and category != 'Others':
                            st.write(f"  {category}: {percentage:.2f}%")
                    
                    if 'Others' in sorted_categories and sorted_categories['Others'] > 0:
                        st.write(f"  Others: {sorted_categories['Others']:.2f}%")
                except FileNotFoundError as e:
                    st.error(f"Error loading model or scaler files: {e}")

if __name__ == "__main__":
    main()
