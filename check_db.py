import sqlite3

conn = sqlite3.connect("database/healthcare.db")
conn.row_factory = sqlite3.Row

print("===== USERS TABLE =====")

users = conn.execute("SELECT * FROM users").fetchall()

print("Total Users:", len(users))

for user in users:
    print(dict(user))

conn.close()