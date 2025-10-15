import streamlit as st
import sqlite3
import pandas as pd

def create_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, email TEXT, age INTEGER)')
    conn.commit()
    conn.close()

def add_user(name, email, age):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO users(name, email, age) VALUES (?, ?, ?)', (name, email, age))
    conn.commit()
    conn.close()

def view_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    data = c.fetchall()
    conn.close()
    return data

def delete_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('DELETE FROM users WHERE id=?', (user_id,))
    conn.commit()
    conn.close()

st.title("User Management App")
st.markdown("---")
create_table()

menu = ["Add User", "View Users", "Delete User"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Add User":
    st.subheader("Add New User")
    name = st.text_input("Name")
    email = st.text_input("Email")
    age = st.number_input("Age", 0, 120)
    st.write("")  # Spacer

    if st.button("Submit"):
        if name and email:
            add_user(name, email, age)
            st.success(f"{name} added successfully!")
        else:
            st.warning("Please fill in all fields.")

elif choice == "View Users":
    st.subheader("View All Users")
    users = view_users()
    df = pd.DataFrame(users, columns=["ID", "Name", "Email", "Age"])
    st.dataframe(df)

elif choice == "Delete User":
    st.subheader("Delete a User")
    users = view_users()
    df = pd.DataFrame(users, columns=["ID", "Name", "Email", "Age"])
    st.dataframe(df)

    if not df.empty:
        user_ids = df["ID"].tolist()
        user_id = st.selectbox("Select ID to delete", user_ids)
        if st.button("Delete"):
            delete_user(user_id)
            st.success(f"User {user_id} deleted!")
    else:
        st.info("No users to delete.")

st.markdown("---")
st.markdown("<div style='text-align: right; color: gray;'>Christine Julliane Reyes</div>", unsafe_allow_html=True)