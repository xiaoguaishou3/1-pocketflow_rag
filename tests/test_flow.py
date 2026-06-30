"""Layer 2: Flow 定义测试"""
from rag_logic.flow import offline_flow, online_flow


class TestOfflineFlow:
    """离线 Flow 测试"""

    def test_flow_exists(self):
        """offline_flow 存在"""
        assert offline_flow is not None

    def test_flow_has_start_node(self):
        """offline_flow 有起始节点"""
        assert hasattr(offline_flow, "start") or hasattr(offline_flow, "_start")


class TestOnlineFlow:
    """在线 Flow 测试"""

    def test_flow_exists(self):
        """online_flow 存在"""
        assert online_flow is not None

    def test_flow_has_start_node(self):
        """online_flow 有起始节点"""
        assert hasattr(online_flow, "start") or hasattr(online_flow, "_start")
