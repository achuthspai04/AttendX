import tkinter as tk
from tkinter import ttk
import sqlite3

class AttendanceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance Table")

        # Create a Treeview widget
        self.tree = ttk.Treeview(root, columns=('Name', 'ID', 'Timestamp'))
        self.tree.heading('#0', text='Index')
        self.tree.heading('#1', text='Name')
        self.tree.heading('#2', text='ID')
        self.tree.heading('#3', text='Timestamp')
        self.tree.pack(expand=True, fill='both')

        # Connect to the database
        self.conn = sqlite3.connect('attendance.db')
        self.cursor = self.conn.cursor()

        # Fetch data from the database and insert into the Treeview
        self.fetch_data()

    def fetch_data(self, query="SELECT * FROM attendances"):
        # Clear existing items in the Treeview
        self.tree.delete(*self.tree.get_children())

        # Fetch data from the database
        self.cursor.execute(query)
        rows = self.cursor.fetchall()

        # Insert data into the Treeview
        for i, row in enumerate(rows):
            self.tree.insert('', 'end', text=i+1, values=row)

    def sort_by_name(self):
        self.fetch_data("SELECT * FROM attendances ORDER BY Name")

    def sort_by_id(self):
    # This explicitly ensures sorting by ID as an integer (useful if IDs are stored as text)
        self.fetch_data("SELECT * FROM attendances ORDER BY CAST(ID as INTEGER)")


    def sort_by_timestamp(self):
        self.fetch_data("SELECT * FROM attendances ORDER BY Time")

root = tk.Tk()
app = AttendanceApp(root)

# Add sorting buttons
sort_by_name_btn = tk.Button(root, text="Sort by Name", command=app.sort_by_name)
sort_by_name_btn.pack(side=tk.LEFT, padx=5, pady=5)

sort_by_id_btn = tk.Button(root, text="Sort by ID", command=app.sort_by_id)
sort_by_id_btn.pack(side=tk.LEFT, padx=5, pady=5)

sort_by_timestamp_btn = tk.Button(root, text="Sort by Timestamp", command=app.sort_by_timestamp)
sort_by_timestamp_btn.pack(side=tk.LEFT, padx=5, pady=5)

root.mainloop()
