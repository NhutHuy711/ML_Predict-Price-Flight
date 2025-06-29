import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Hàm tiền xử lý dữ liệu mới (từ notebook)
def preprocess_new_data(new_data_raw, df_train, cat_cols):
    # Rút gọn 'days_left' giống như trong tập huấn luyện streamlit run app.py
    conditions = [
        new_data_raw['days_left'].between(1, 5),
        new_data_raw['days_left'].between(6, 10),
        new_data_raw['days_left'].between(11, 15),
        new_data_raw['days_left'] > 15
    ]
    choices = [new_data_raw['days_left'], 6, 11, 15]
    new_data_raw['days_left'] = np.select(conditions, choices, default=0)

    # One-hot encoding cho các cột phân loại
    new_data_encoded = pd.get_dummies(new_data_raw, columns=cat_cols, drop_first=True)

    # Đảm bảo các cột giống với tập huấn luyện
    model_cols = df_train.drop(columns=['price']).columns
    for col in model_cols:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0

    # Đảm bảo đúng thứ tự cột
    new_data_encoded = new_data_encoded[model_cols]
    return new_data_encoded

# Tạo giao diện Streamlit
st.title("Dự đoán giá vé máy bay")
st.write("Nhập thông tin chuyến bay để dự đoán giá vé.")

# Tạo các trường nhập liệu
airline = st.selectbox("Hãng hàng không", ["SpiceJet", "Air_India", "Indigo", "GO_FIRST", "Vistara", "AirAsia"])
source_city = st.selectbox("Thành phố khởi hành", ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
departure_time = st.selectbox("Thời gian khởi hành", ["Evening", "Early_Morning", "Morning", "Night", "Afternoon", "Late_Night"])
stops = st.selectbox("Số điểm dừng", ["zero", "one", "two_or_more"])
arrival_time = st.selectbox("Thời gian đến", ["Night", "Evening", "Morning", "Afternoon", "Early_Morning", "Late_Night"])
destination_city = st.selectbox("Thành phố đến", ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad", "Chennai"])
class_type = st.selectbox("Hạng ghế", ["Economy", "Business"])
duration = st.number_input("Thời gian bay (giờ)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
days_left = st.number_input("Số ngày còn lại đến chuyến bay", min_value=1, max_value=49, value=30, step=1)

# Danh sách các cột phân loại
cat_cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

# Tạo DataFrame cho dữ liệu nhập vào
new_data = pd.DataFrame({
    'airline': [airline],
    'source_city': [source_city],
    'departure_time': [departure_time],
    'stops': [stops],
    'arrival_time': [arrival_time],
    'destination_city': [destination_city],
    'class': [class_type],
    'duration': [duration],
    'days_left': [days_left]
})

# Tạo DataFrame giả lập df_train với các cột giống dataset
df_train = pd.DataFrame(columns=[
    'duration', 'days_left', 'price', 'airline_Air_India', 'airline_GO_FIRST', 'airline_Indigo',
    'airline_SpiceJet', 'airline_Vistara', 'source_city_Chennai', 'source_city_Delhi',
    'source_city_Hyderabad', 'source_city_Kolkata', 'source_city_Mumbai',
    'departure_time_Early_Morning', 'departure_time_Evening', 'departure_time_Late_Night',
    'departure_time_Morning', 'departure_time_Night', 'stops_two_or_more', 'stops_zero',
    'arrival_time_Early_Morning', 'arrival_time_Evening', 'arrival_time_Late_Night',
    'arrival_time_Morning', 'arrival_time_Night', 'destination_city_Chennai',
    'destination_city_Delhi', 'destination_city_Hyderabad', 'destination_city_Kolkata',
    'destination_city_Mumbai', 'class_Economy'
])

# Nút dự đoán
if st.button("Dự đoán giá vé"):
    try:
        # Tiền xử lý dữ liệu
        new_data_preprocessed = preprocess_new_data(new_data, df_train, cat_cols)

        # Tải mô hình đã lưu
        final_model = joblib.load('/Machine Learning/Final_Model.pkl')

        # Dự đoán giá
        predicted_price = final_model.predict(new_data_preprocessed)

        # Hiển thị kết quả
        st.success(f"Giá vé dự đoán: {predicted_price[0]:,.2f}")
    except FileNotFoundError:
        st.error("Không tìm thấy file mô hình 'Final_Model.pkl'. Vui lòng đảm bảo file này tồn tại trong thư mục hiện tại.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi: {str(e)}")

# Run in terminal:
# streamlit run "/Users/lephu/Documents/Code/Pycharm/DataScience/Machine Learning/app.py"