import pybullet as p

class ZeetaPhysics:
    """
    A wrapper for the PyBullet physics client to be passed to other Zeeta modules.
    """
    def __init__(self, client_id: int):
        self._client = client_id
        self._pb = p

    @property
    def client(self) -> int:
        return self._client

    @property
    def pb(self):
        return self._pb
