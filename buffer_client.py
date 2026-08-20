import zmq
import json
import numpy as np
import time

class BufferClient:
    """
    Client interface for interacting with the Node Ingestion and Buffer Services.
    Allows real-time streaming subscription and historical reach-back queries.
    """
    def __init__(self, bind_address, pub_port, rep_port, security_token):
        self.bind_address = bind_address
        self.pub_port = pub_port
        self.rep_port = rep_port
        self.security_token = security_token
        
        self.context = zmq.Context()
        self.sub_socket = None
        self.rep_socket = None

    def connect_subscriber(self):
        """
        Connects to the real-time live publisher socket.
        """
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_socket.setsockopt(zmq.RCVHWM, 1000)
        self.sub_socket.connect(f"tcp://{self.bind_address}:{self.pub_port}")

    def receive_live_frame(self, timeout_ms=1000):
        """
        Retrieves a single live frame (metadata + data numpy array) from subscription feed.
        """
        if self.sub_socket is None:
            self.connect_subscriber()
            
        poller = zmq.Poller()
        poller.register(self.sub_socket, zmq.POLLIN)
        
        socks = dict(poller.poll(timeout=timeout_ms))
        if self.sub_socket in socks:
            metadata = self.sub_socket.recv_json()
            raw_data = self.sub_socket.recv()
            
            # Security token verification
            if metadata.get("token") != self.security_token:
                raise PermissionError("Security mismatch: invalid token received")
                
            shape = tuple(metadata["shape"])
            iq_data = np.frombuffer(raw_data, dtype=np.complex64).reshape(shape)
            return metadata, iq_data
            
        return None, None

    def query_history(self, start_time, end_time, frequency_hz=None, timeout_ms=5000):
        """
        Sends query request to the buffer service to fetch historical data slices.
        """
        if self.rep_socket is None:
            self.rep_socket = self.context.socket(zmq.REQ)
            self.rep_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            self.rep_socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            self.rep_socket.connect(f"tcp://{self.bind_address}:{self.rep_port}")
            
        query = {
            "token": self.security_token,
            "start_time": start_time,
            "end_time": end_time,
            "frequency_hz": frequency_hz
        }
        
        try:
            self.rep_socket.send_json(query)
            
            # Receive response metadata
            response_meta = self.rep_socket.recv_json()
            if "error" in response_meta:
                raise RuntimeError(f"Server error: {response_meta['error']}")
                
            raw_data = self.rep_socket.recv()
            
            match_count = response_meta["match_count"]
            if match_count > 0:
                # Matched data shape will be (match_count, M, N)
                # First get the shape of a single frame from the first match
                matches = response_meta["matches"]
                shape = (match_count, 5, 1024) # standard shape
                matched_data = np.frombuffer(raw_data, dtype=np.complex64).reshape(shape)
                return response_meta, matched_data
            else:
                return response_meta, np.array([], dtype=np.complex64)
                
        except zmq.error.Again:
            # Recreate socket on timeout to reset the state machine
            self.rep_socket.close()
            self.rep_socket = None
            raise TimeoutError("Buffer server did not respond to historical query.")

    def close(self):
        if self.sub_socket:
            self.sub_socket.close()
        if self.rep_socket:
            self.rep_socket.close()
        self.context.term()
