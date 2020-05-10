import pymysql

class ConnectionConfig:
    def connection(self, user="sql7339253", password="P6ajnRvgAW"):
        self.conn = pymysql.connect("sql7.freemysqlhosting.net", user, password, "sql7339253")
        if(self.conn):
            print("...connect with database...")
        else:
            print("wrong connection")
        return self.conn
    def closeConnection(self):
        self.conn.close()
        print("...connection closed...")
#
#
# ConnectionConfig().connection()