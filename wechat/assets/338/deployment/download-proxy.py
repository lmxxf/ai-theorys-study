import socket, socketserver, select
class Proxy(socketserver.StreamRequestHandler):
 def handle(self):
  line=self.rfile.readline().decode().split()
  if len(line)<2:return
  while self.rfile.readline() not in (b"\r\n",b"\n",b""):pass
  if line[0]!="CONNECT":
   self.wfile.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n");return
  host,port=line[1].rsplit(":",1)
  try: remote=socket.create_connection((host,int(port)),30)
  except OSError:
   self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n");return
  self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n");self.wfile.flush()
  with remote:
   while True:
    ready,_,_=select.select([self.connection,remote],[],[],120)
    if not ready:return
    for src in ready:
     data=src.recv(262144)
     if not data:return
     (remote if src is self.connection else self.connection).sendall(data)
class Server(socketserver.ThreadingTCPServer):
 allow_reuse_address=True
 daemon_threads=True
Server(("127.0.0.1",17899),Proxy).serve_forever()
