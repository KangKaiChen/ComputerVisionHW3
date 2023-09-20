import math
import cv2
import numpy as np
import matplotlib.pyplot as plt



def rgb_to_grayscale(img):
    gray_img = np.zeros(img.shape[:2])
    for i in range(gray_img.shape[0]):
        for j in range(gray_img.shape[1]):
            gray_img[i][j] = int(0.299 * img[i][j][0] +
                                 0.587 * img[i][j][1] + 0.114 * img[i][j][2])
    return gray_img


def create_gaussian_kernel(size=3, sigma=1):
    kernel = np.zeros([size, size])
    sum = 0
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            x = int(-(kernel.shape[0] - 1) * 0.5 + i)
            y = int(-(kernel.shape[0] - 1) * 0.5 + j)
            intensity = math.exp(-(x**2 + y**2) /
                                 (2 * sigma**2)) / (2 * math.pi * sigma**2)
            sum += intensity  # for normalization
            kernel[i][j] = intensity

    return kernel / sum


def convolution(img, kernel):
    output_img = np.zeros(img.shape)
    padded_img = np.pad(img, int((kernel.shape[0] - 1) * 0.5), "edge")

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for m in range(kernel.shape[0]):
                for n in range(kernel.shape[1]):
                    output_img[i][j] += padded_img[i + m][j + n] * kernel[m][n]

    return output_img


def cal_grad_magnitude(grad_x, grad_y):
    magnitude = np.zeros(grad_x.shape)
    for i in range(grad_x.shape[0]):
        for j in range(grad_x.shape[1]):
            magnitude[i][j] = math.sqrt(grad_x[i][j] ** 2 + grad_y[i][j] ** 2)

    return magnitude


def cal_grad_angle(grad_x, grad_y):
    angle = np.zeros(grad_x.shape)
    for i in range(grad_x.shape[0]):
        for j in range(grad_x.shape[1]):
            if grad_y[i][j] != 0:
                angle[i][j] = math.atan(grad_x[i][j] / grad_y[i][j])
            else:
                angle[i][j] = math.atan(grad_x[i][j] / 0.00001)

    return angle


def non_maximal_suppression(magnitude, angle, kernel_size=3):
    for i in range(magnitude.shape[0] - kernel_size + 1):
        for j in range(magnitude.shape[1] - kernel_size + 1):
            neighbors = magnitude[i: i + kernel_size, j: j + kernel_size]
            max_index = np.unravel_index(neighbors.argmax(), neighbors.shape)
            max_angle = angle[i + max_index[0]][j + max_index[1]]
            if max_angle < 0:
                max_angle += math.pi

            if max_angle < math.pi / 4 or max_angle > math.pi * 3 / 4:
                candidate = [idx for idx in range(kernel_size)]
                candidate.remove(max_index[0])
                for idx in candidate:
                    magnitude[i + idx][j] = 0
            else:
                candidate = [idx for idx in range(kernel_size)]
                candidate.remove(max_index[1])
                for idx in candidate:
                    magnitude[i][j + idx] = 0


def double_threshold(magnitude, high_thred_ratio=0.12, low_thred_ratio=0.1):
    padded_grid = np.pad(magnitude, 1)

    high_thred = magnitude.max() * high_thred_ratio
    low_thred = magnitude.max() * low_thred_ratio

    for i in range(padded_grid.shape[0]):
        for j in range(padded_grid.shape[1]):
            if padded_grid[i][j] > high_thred:
                padded_grid[i][j] = 255
            elif padded_grid[i][j] < low_thred:
                padded_grid[i][j] = 0

    for i in range(magnitude.shape[0]):
        for j in range(magnitude.shape[1]):
            if padded_grid[i + 1][j + 1] == 255:
                magnitude[i][j] = 255
            elif padded_grid[i + 1][j + 1] == 0:
                magnitude[i][j] = 0
            else:
                magnitude[i][j] = 255 if 255 in padded_grid[i: i +
                                                            3, j: j + 3] else 0


def hough_transform(edges, theta_resolution, rho_resolution, threshold):
    height, width = edges.shape
    max_rho = math.ceil(math.sqrt(height**2 + width**2))
    theta_values = np.deg2rad(np.arange(0, 180, theta_resolution))
    num_thetas = len(theta_values)
    num_rhos = int(2 * max_rho / rho_resolution + 1)
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.uint64)

    for y in range(height):
        for x in range(width):
            if edges[y, x] != 0:
                for theta_index in range(num_thetas):
                    theta = theta_values[theta_index]
                    rho = x * math.cos(theta) + y * math.sin(theta)
                    rho_index = int(rho + max_rho)
                    accumulator[rho_index, theta_index] += 1

    lines = []
    for rho_index in range(num_rhos):
        for theta_index in range(num_thetas):
            if accumulator[rho_index, theta_index] >= threshold:
                rho = rho_index - max_rho
                theta = theta_values[theta_index]
                lines.append((rho, theta))

    return lines

# 假設您有一個名為"image"的二值化圖像


def plot_hough_lines(image, accumulator, theta_range, threshold):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image, cmap='gray')

    # 根據累加器中的值，選擇高於閾值的直線
    lines = []
    rhos, thetas = np.nonzero(accumulator >= threshold)
    for rho, theta_idx in zip(rhos, thetas):
        theta = theta_range[theta_idx]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        lines.append(((x1, y1), (x2, y2)))


    # 繪製檢測到的直線
    for line in lines:
        (x1, y1), (x2, y2) = line
     #   ax.plot([x1, x2], [y1, y2], color='red', linewidth=2)
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


   # plt.show()


if __name__ == "__main__":

    test_image_filename = "test_img/4.jpg"
    output_Gaussian_filename = "4_Gaussian.jpg"
    output_image_filename = "4_canny.jpg"
    output_hough_filename = "4_hough.jpg"
    test_image = cv2.imread(test_image_filename)
    if not isinstance(test_image, np.ndarray):
        print(f'Image "{test_image_filename}" not found!')
        exit()
    print(f'image "{test_image_filename}" loaded. size: {test_image.shape}')

    test_image_gray = rgb_to_grayscale(test_image)
    # cv2.imwrite(f"{test_image_filename[:-4]}_gray.jpg", test_image_gray)

    gaussian_kernel = create_gaussian_kernel(3)
    smoothed_image = convolution(test_image_gray, gaussian_kernel)

    cv2.imwrite(output_Gaussian_filename, smoothed_image)

    sobelx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolution(smoothed_image, sobelx_kernel)
    grad_y = convolution(smoothed_image, sobely_kernel)
    # cv2.imwrite(f"{test_image_filename[:-4]}_gradx.jpg", grad_x)
    # cv2.imwrite(f"{test_image_filename[:-4]}_grady.jpg", grad_y)

    grad_magnitude = cal_grad_magnitude(grad_x, grad_y)
    # cv2.imwrite(f"{test_image_filename[:-4]}_grad.jpg", grad_magnitude_grid)

    grad_angle = cal_grad_angle(grad_x, grad_y)

    non_maximal_suppression(grad_magnitude, grad_angle)
    # cv2.imwrite(f"{test_image_filename[:-4]}_grad_thin.jpg", grad_magnitude_grid)

    double_threshold(grad_magnitude)
    theta_resolution = 1  # 角度解析度（以度為單位）
    rho_resolution = 1  # 距離解析度（以像素為單位）
    threshold = 130 # 投票閾值

    lines = hough_transform(grad_magnitude, theta_resolution, rho_resolution, threshold)
    result = test_image.copy()
    for line in lines:
        rho, theta = line
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)


    cv2.imshow("hough tansform ", result)
    cv2.imwrite(output_hough_filename, result)


    cv2.imwrite(output_image_filename, grad_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()