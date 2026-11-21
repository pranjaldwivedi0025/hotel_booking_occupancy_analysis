# PROJECT : Hotel Booking & Occupancy Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random

# ---------- 1. Dataset Banana (Real jaisa) ----------
np.random.seed(13)
n = 800
hotels = ['Taj Palace Delhi', 'Oberoi Mumbai', 'ITC Grand Goa', 'Leela Jaipur', 'Marriott Bengaluru', 'Hyatt Hyderabad']
cities = ['Delhi', 'Mumbai', 'Goa', 'Jaipur', 'Bengaluru', 'Hyderabad']
room_types = ['Standard', 'Deluxe', 'Suite', 'Executive']
channels = ['OTA', 'Direct', 'Travel Agent']
guests = ['Couple', 'Family', 'Business', 'Solo']

data = {
    'Booking_ID': range(1001, 1001+n),
    'Hotel_Name': [random.choice(hotels) for _ in range(n)],
    'City': [random.choice(cities) for _ in range(n)],
    'Room_Type': [random.choice(room_types) for _ in range(n)],
    'Booking_Channel': [random.choice(channels) for _ in range(n)],
    'Guest_Type': [random.choice(guests) for _ in range(n)],
    'Booking_Date': [datetime(2024,1,1) + timedelta(days=random.randint(0,365)) for _ in range(n)],
    'Stay_Date': [datetime(2024,1,1) + timedelta(days=random.randint(0,365)) for _ in range(n)],
    'Stay_Length': np.random.randint(1, 8, n),
    'Room_Price': np.random.choice([2500, 3500, 4500, 6000, 8500, 12000, 18000], n),
    'Booking_Status': [random.choice(['Confirmed','Cancelled','No-Show']) for _ in range(n)],
    'Total_Amount': [0]*n
}
df = pd.DataFrame(data)
# Real jaisa missing data daal diya
df.loc[5, 'Room_Price'] = np.nan
df.loc[12, 'Guest_Type'] = np.nan
# Total Amount = Price * Stay Length
df['Total_Amount'] = df['Room_Price'] * df['Stay_Length']
df.to_csv('13_hotel_bookings.csv', index=False)

print("1. Shape:", df.shape)
print(df.head())
print(df.dtypes)

# ---------- 2. Missing + Duplicate Check ----------
print("\n2. Missing Values:\n", df.isnull().sum())
print("Duplicate Booking_ID:", df.duplicated(subset=['Booking_ID']).sum())

# ---------- 3. Handle Missing Values ----------
df['Room_Price'].fillna(df['Room_Price'].median(), inplace=True)
df['Guest_Type'].fillna(df['Guest_Type'].mode()[0], inplace=True)
df['Total_Amount'] = df['Room_Price'] * df['Stay_Length']
print("\nAfter Cleaning Missing:", df.isnull().sum().sum())

# ---------- 4. Datetime + Month, Weekday ----------
df['Booking_Date'] = pd.to_datetime(df['Booking_Date'])
df['Stay_Date'] = pd.to_datetime(df['Stay_Date'])
df['Booking_Month'] = df['Booking_Date'].dt.month_name()
df['Booking_Month_Num'] = df['Booking_Date'].dt.month
df['Stay_Weekday'] = df['Stay_Date'].dt.day_name()
df['Booking_Year'] = df['Booking_Date'].dt.year

# ---------- 5. Total Bookings, Avg Price, Occupancy, Cancellation % ----------
total_bookings = len(df)
avg_price = df['Room_Price'].mean()
avg_stay = df['Stay_Length'].mean()
cancellation_pct = (df['Booking_Status']=='Cancelled').mean()*100
total_revenue = df[df['Booking_Status']=='Confirmed']['Total_Amount'].sum()
print(f"\n5. Total Bookings: {total_bookings}")
print(f"Avg Room Price: Rs {avg_price:.0f}")
print(f"Avg Stay Length: {avg_stay:.2f} nights")
print(f"Cancellation %: {cancellation_pct:.2f}%")
print(f"Total Revenue (Confirmed Only): Rs {total_revenue:.0f}")

# ---------- 6. Top 10 Cities / Hotels ----------
print("\n6. Top Hotels by Bookings:\n", df['Hotel_Name'].value_counts().head(10))
print("\nTop Cities:\n", df['City'].value_counts())

# ---------- 7. Rank Room Types by Demand & Revenue ----------
room_rank = df.groupby('Room_Type').agg(Bookings=('Booking_ID','count'), Revenue=('Total_Amount','sum'), Avg_Price=('Room_Price','mean')).sort_values(by='Bookings', ascending=False)
print("\n7. Room Type Ranking:\n", room_rank)

# ---------- 8. Booking Channel Comparison ----------
channel_comp = df.groupby('Booking_Channel').agg(Bookings=('Booking_ID','count'), Avg_Price=('Room_Price','mean'), Revenue=('Total_Amount','sum')).sort_values(by='Bookings', ascending=False)
print("\n8. Channel Comparison:\n", channel_comp)

# ---------- 9. Cancellation by Room Type & Channel ----------
print("\n9. Cancellation % by Room Type:\n", df.groupby('Room_Type')['Booking_Status'].apply(lambda x: (x=='Cancelled').mean()*100))
print("\nCancellation % by Channel:\n", df.groupby('Booking_Channel')['Booking_Status'].apply(lambda x: (x=='Cancelled').mean()*100))

# ---------- 10. List / Tuple / Set / Dict ----------
room_list = list(df['Room_Type'].unique())
city_tuple = tuple(df['City'].unique())
channel_set = set(df['Booking_Channel'].unique())
city_revenue_dict = df.groupby('City')['Total_Amount'].sum().to_dict()
print("\n10. List:", room_list)
print("Tuple Cities:", city_tuple)
print("Set Channels:", channel_set)
print("Dict City Revenue:", city_revenue_dict)

# ---------- 11. Function calculate_total_price() ----------
def calculate_total_price(room_price, stay_length, discount_pct=0):
    total = room_price * stay_length
    discount = total * (discount_pct/100)
    return total - discount

print("\n11. Function Example: 5000 x 3 nights, 10% off =", calculate_total_price(5000, 3, 10))

# ---------- 12. NumPy Mean, Median, Std, Percentile ----------
print("\n12. NumPy Stats - Room Price:")
print("Mean:", np.mean(df['Room_Price']), "Median:", np.median(df['Room_Price']), "Std:", np.std(df['Room_Price']))
print("Percentiles 25,50,75,90:", np.percentile(df['Room_Price'], [25,50,75,90]))
print("Stay Length - Mean:", np.mean(df['Stay_Length']), "Median:", np.median(df['Stay_Length']), "Std:", np.std(df['Stay_Length']))

# ---------- 13. IQR Outlier for Room_Price ----------
Q1 = np.percentile(df['Room_Price'], 25)
Q3 = np.percentile(df['Room_Price'], 75)
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers = df[(df['Room_Price'] < lower) | (df['Room_Price'] > upper)]
print(f"\n13. Outliers in Room Price: {len(outliers)} | Upper Limit: Rs {upper:.0f}")

# ---------- 14. pd.cut() - Price Category ----------
df['Price_Category'] = pd.cut(df['Room_Price'], bins=[0,4000,8000,20000], labels=['Budget','Mid-Range','Premium'])
print("\nPrice Category Wise:\n", df.groupby('Price_Category', observed=True).size())

# ---------- 15. Class HotelBooking ----------
class HotelBooking:
    def __init__(self, booking_id, hotel, room_type, price, stay_length, status):
        self.booking_id = booking_id
        self.hotel = hotel
        self.room_type = room_type
        self.price = price
        self.stay_length = stay_length
        self.status = status
    def total_price(self):
        return self.price * self.stay_length
    def is_cancelled(self):
        return self.status == 'Cancelled'

b1 = HotelBooking(1001, 'Taj Palace Delhi', 'Deluxe', 6000, 3, 'Confirmed')
print("\nClass Example:", b1.hotel, "Total:", b1.total_price(), "Cancelled?", b1.is_cancelled())

# ---------- 16. Visuals ----------
plt.figure(figsize=(10,4))
monthly = df.groupby('Booking_Month_Num')['Booking_ID'].count()
sns.lineplot(x=monthly.index, y=monthly.values, marker='o')
plt.title('Monthly Booking Trend')
plt.xlabel('Month'); plt.ylabel('Bookings')
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(data=df, x='Booking_Channel', order=df['Booking_Channel'].value_counts().index)
plt.title('Bookings by Channel (OTA vs Direct vs Agent)')
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(data=df, x='Room_Type', y='Total_Amount', estimator='sum')
plt.title('Revenue by Room Type')
plt.show()

plt.figure(figsize=(8,4))
sns.countplot(data=df, x='Booking_Status')
plt.title('Booking Status - Confirmed vs Cancelled')
plt.show()

# Save cleaned file
df.to_csv('13_hotel_cleaned.csv', index=False)
print("\nCleaned file saved: 13_hotel_cleaned.csv")

print("""
RECOMMENDATIONS:
1. OTA se sabse zyada booking aati hai - OTA par offers aur photos achhe rakho, par Direct booking badhane ke liye website par 10% discount do taaki commission bache.
2. Deluxe room sabse zyada demand me hai - uska stock aur promotion badhao, Suite ka price thoda kam karke occupancy badhao.
3. Cancellation Travel Agent channel me sabse zyada hai - Agent ke liye strict cancellation policy aur advance payment lagao.
4. Weekend aur December jaise peak months me price thoda badhao (dynamic pricing) aur weekdays me discount do taaki occupancy stable rahe.
5. Premium category me revenue zyada hai par bookings kam hai - Business guests ke liye early check-in / late check-out jaise perks do.
""")