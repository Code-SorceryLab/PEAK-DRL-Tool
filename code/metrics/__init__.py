try:
    from .flappy_balance import FlappyBalanceStats  # noqa: F401 – export default collector
except ImportError:
    pass  # flappy_balance module is optional; not present in all configurations