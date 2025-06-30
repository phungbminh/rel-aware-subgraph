import logging
import os

def setup_logger(log_path=None, level=logging.INFO):
    """
    Thiết lập logger ghi log ra cả console và file (nếu chỉ định).
    - log_path: Đường dẫn file log (vd: 'logs/experiment.log'), None nếu chỉ log ra console.
    - level: logging level (mặc định: INFO)
    Return:
        logger object (dùng .info(), .warning()...)
    """
    logger = logging.getLogger('RASG')
    logger.setLevel(level)
    logger.propagate = False  # Ngăn log bị nhân đôi khi import nhiều lần

    # Xoá tất cả handler cũ (tránh trùng lặp log khi chạy lại notebook/script)
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler (log ra màn hình)
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s',
                                       datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (log ra file, nếu có log_path)
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger
