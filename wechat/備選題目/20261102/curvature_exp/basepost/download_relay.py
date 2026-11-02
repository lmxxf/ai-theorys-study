#!/usr/bin/env python3
"""Temporary loopback-only CONNECT relay for public HF model downloads. No GPU use."""
import socket,socketserver,select
class Handler(socketserver.StreamRequestHandler):
 def handle(self):
  try:
   line=self.rfile.readline(8192).decode('ascii');method,target,_=line.split()
   while self.rfile.readline(8192) not in (b'\r\n',b'\n',b''):pass
   host,port=target.rsplit(':',1)
   allowed=host=='huggingface.co' or host.endswith(('.huggingface.co','.hf.co','.xethub.hf.co'))
   if method!='CONNECT' or port!='443' or not allowed:
    self.wfile.write(b'HTTP/1.1 403 Forbidden\r\n\r\n');return
   upstream=socket.create_connection((host,int(port)),timeout=15)
   with upstream:
    upstream.settimeout(None);self.connection.settimeout(None)
    self.wfile.write(b'HTTP/1.1 200 Connection Established\r\n\r\n');self.wfile.flush()
    pair=[self.connection,upstream]
    while True:
     ready,_,_=select.select(pair,[],[],90)
     if not ready:return
     for source in ready:
      data=source.recv(256*1024)
      if not data:return
      (upstream if source is self.connection else self.connection).sendall(data)
  except (OSError,ValueError):return
class Server(socketserver.ThreadingTCPServer):
 allow_reuse_address=True
 daemon_threads=True
with Server(('127.0.0.1',18891),Handler) as server:server.serve_forever()
