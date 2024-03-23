import numpy as np
import cv2
import select
import socket


def create_gamma_img(gamma, img):
  gamma_cvt = np.zeros((256, 1), dtype=np.uint8)
  for i in range(256):
    gamma_cvt[i][0] = 255*(float(i)/255)**(1.0/gamma)
  return cv2.LUT(img, gamma_cvt)


def clear_socket_buffer(conn: socket.socket, timeout):
  while True:
    ready = select.select([conn], [], [], timeout)
    if ready[0]:
      data = conn.recv(1024 * 1024 * 3)
    else:
      return