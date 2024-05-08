import sqlite3
self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendances'")
if self.cur.fetchone():
    print("Table 'attendances' exists")
else:
    print("Table 'attendances' does not exist")
