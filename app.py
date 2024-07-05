import streamlit as st
import cv2
import face_recognition
import sqlite3
import numpy as np

DATABASE = 'faces.db'
USERS_DATABASE = 'users.db'

# Initialize the databases
def get_db(database):
    conn = sqlite3.connect(database)
    return conn

def init_db():
    conn = get_db(DATABASE)
    with conn:
        conn.execute('CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, mobile TEXT, encoding BLOB, image BLOB)')
    conn.close()

    conn = get_db(USERS_DATABASE)
    with conn:
        conn.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)')
    conn.close()

def update_db_schema():
    conn = get_db(DATABASE)
    with conn:
        # Check if the 'image' column exists
        c = conn.cursor()
        c.execute("PRAGMA table_info(faces)")
        columns = [col[1] for col in c.fetchall()]
        if 'image' not in columns:
            conn.execute('ALTER TABLE faces ADD COLUMN image BLOB')
    conn.close()

def load_known_faces():
    conn = get_db(DATABASE)
    c = conn.cursor()
    known_face_encodings = []
    known_face_names = []
    known_face_mobiles = []
    for row in c.execute('SELECT name, mobile, encoding FROM faces'):
        known_face_names.append(row[0])
        known_face_mobiles.append(row[1])
        known_face_encodings.append(np.frombuffer(row[2], dtype=np.float64))
    conn.close()
    return known_face_names, known_face_mobiles, known_face_encodings

def save_new_face(encoding, image, name, mobile):
    encoding_str = sqlite3.Binary(encoding.tobytes())
    image_str = sqlite3.Binary(image)
    conn = get_db(DATABASE)
    c = conn.cursor()
    known_encodings = [np.frombuffer(row[0], dtype=np.float64) for row in c.execute('SELECT encoding FROM faces')]
    if not any(np.allclose(known_encoding, encoding) for known_encoding in known_encodings):
        c.execute('INSERT INTO faces (name, mobile, encoding, image) VALUES (?, ?, ?, ?)', (name, mobile, encoding_str, image_str))
        conn.commit()
        print("New face found. Encoding saved to database.")
    conn.close()

def detect_and_display(frame, known_face_encodings, known_face_names, known_face_mobiles):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    faces_in_frame = set()
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        match_found = False
        for known_face_encoding, name, mobile in zip(known_face_encodings, known_face_names, known_face_mobiles):
            matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
            if True in matches:
                match_found = True
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f'{name}, {mobile}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break
        
        if not match_found and tuple(face_encoding) not in faces_in_frame:
            faces_in_frame.add(tuple(face_encoding))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, 'New Face', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return frame, face_locations, face_encodings

def login(username, password):
    conn = get_db(USERS_DATABASE)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = c.fetchone()
    conn.close()
    if user:
        return True
    else:
        return False

def signup(username, password):
    conn = get_db(USERS_DATABASE)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    if c.fetchone():
        conn.close()
        return False
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
    conn.commit()
    conn.close()
    return True

init_db()
update_db_schema()  # Ensure the database schema is up to date
known_face_names, known_face_mobiles, known_face_encodings = load_known_faces()

# Streamlit application
st.title("Face Recognition")

# Login
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    choice = st.sidebar.selectbox("Login/Signup", ["Login", "Signup"])
    
    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Login"):
            if login(username, password):
                st.session_state['logged_in'] = True
                st.success("Logged in successfully")
                st.experimental_rerun()  # Rerun the app to go to the main page
            else:
                st.error("Invalid username or password")
    
    elif choice == "Signup":
        username = st.text_input("Username")
        password = st.text_input("Password", type='password')
        if st.button("Signup"):
            if signup(username, password):
                st.success("Signup successful. Please login.")
            else:
                st.error("Username already exists")

else:
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Face Recognition", "Manage Faces", "Add Face", "Delete Face"])

    if page == "Face Recognition":
        st.header("Face Recognition")
        
        run = st.checkbox("Run")
        FRAME_WINDOW = st.image([])

        video_capture = cv2.VideoCapture(0)
        
        while run:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            frame, _, _ = detect_and_display(frame, known_face_encodings, known_face_names, known_face_mobiles)
            FRAME_WINDOW.image(frame, channels="BGR")
        
        video_capture.release()

    elif page == "Manage Faces":
        st.header("Manage Faces")
        
        conn = get_db(DATABASE)
        c = conn.cursor()
        search_by = st.radio("Search by", ["Name", "Mobile"])
        search_value = st.text_input(f"Enter {search_by}")
        query = f"SELECT id, name, mobile, image FROM faces WHERE {search_by.lower()} LIKE ?"
        c.execute(query, (f'%{search_value}%',))
        faces = c.fetchall()
        conn.close()
        
        if faces:
            for face in faces:
                st.write(f"Name: {face[1]}, Mobile: {face[2]}")
                image = face[3]
                if image:
                    st.image(image, channels="BGR")
                if st.button(f"Edit {face[0]}", key=f"edit_{face[0]}"):
                    st.session_state[f"editing_{face[0]}"] = True
                if st.session_state.get(f"editing_{face[0]}", False):
                    new_name = st.text_input("Enter new name", face[1], key=f"name_{face[0]}")
                    new_mobile = st.text_input("Enter new mobile", face[2], key=f"mobile_{face[0]}")
                    if st.button("Save", key=f"save_{face[0]}"):
                        conn = get_db(DATABASE)
                        c = conn.cursor()
                        c.execute('UPDATE faces SET name = ?, mobile = ? WHERE id = ?', (new_name, new_mobile, face[0]))
                        conn.commit()
                        conn.close()
                        st.success("Updated successfully")
                        st.session_state[f"editing_{face[0]}"] = False
                        st.experimental_rerun()

    elif page == "Add Face":
        st.header("Add Face")

        video_capture = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])

        if st.button("Capture"):
            ret, frame = video_capture.read()
            if ret:
                st.image(frame, channels="BGR")
                face_locations = face_recognition.face_locations(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if face_locations:
                    face_encodings = face_recognition.face_encodings(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), face_locations)
                    st.session_state['captured_face_encoding'] = face_encodings[0]
                    st.session_state['captured_frame'] = frame
                    st.success("Face captured successfully")
                else:
                    st.error("No face detected. Try again.")
        
        if 'captured_face_encoding' in st.session_state:
            name = st.text_input("Enter name")
            mobile = st.text_input("Enter mobile")
            if st.button("Save"):
                save_new_face(st.session_state['captured_face_encoding'], cv2.imencode('.jpg', st.session_state['captured_frame'])[1].tobytes(), name, mobile)
                st.success("Face added successfully")
                del st.session_state['captured_face_encoding']
                del st.session_state['captured_frame']
        
        video_capture.release()

    elif page == "Delete Face":
        st.header("Delete Face")
        
        conn = get_db(DATABASE)
        c = conn.cursor()
        search_by = st.radio("Search by", ["Name", "Mobile"])
        search_value = st.text_input(f"Enter {search_by}")
        query = f"SELECT id, name, mobile, image FROM faces WHERE {search_by.lower()} LIKE ?"
        c.execute(query, (f'%{search_value}%',))
        faces = c.fetchall()
        conn.close()
        
        if faces:
            for face in faces:
                st.write(f"Name: {face[1]}, Mobile: {face[2]}")
                image = face[3]
                if image:
                    st.image(image, channels="BGR")
                delete = st.button(f"Delete {face[0]}", key=f"delete_{face[0]}")
                
                if delete:
                    st.session_state[f"confirm_delete_{face[0]}"] = True
                if st.session_state.get(f"confirm_delete_{face[0]}", False):
                    if st.button("Confirm Delete", key=f"confirm_{face[0]}"):
                        conn = get_db(DATABASE)
                        c = conn.cursor()
                        c.execute('DELETE FROM faces WHERE id = ?', (face[0],))
                        conn.commit()
                        conn.close()
                        st.success("Deleted successfully")
                        del st.session_state[f"confirm_delete_{face[0]}"]
                        st.experimental_rerun()
