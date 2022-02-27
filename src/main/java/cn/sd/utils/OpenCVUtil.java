package cn.sd.utils;


import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.objdetect.CascadeClassifier;

import java.net.URL;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.*;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_COLOR;
import static org.opencv.imgcodecs.Imgcodecs.imread;

public class OpenCVUtil {

    static String classifierFile = "src/main/resources/lib/haarcascades/haarcascade_frontalface_alt2.xml";

    public static ArrayList<String> findAndCutFace(String path, String savePath) {
        ArrayList<String> cutFaceSavePosition = new ArrayList<>();
        // 解决awt报错问题
        System.setProperty("java.awt.headless", "false");
        // System.out.println(System.getProperty("java.library.path"));

        // 加载动态库
        URL url = ClassLoader.getSystemResource("lib/opencv_java343.dll");
        CascadeClassifier faceDetector = new CascadeClassifier();
        boolean loadSuccess = faceDetector.load(classifierFile);
        if (loadSuccess) {
            System.out.println("加载分类器成功:" + loadSuccess);
            // 读取图像
            Mat image = imread(path);
            // 用于保存检测到的人脸
            MatOfRect faceDetections = new MatOfRect();
            // 开始检测人脸
            faceDetector.detectMultiScale(image, faceDetections, 1.1, 3, 0, new Size(20, 20));
            // 检测到的人脸矩形坐标
            Rect[] faces = faceDetections.toArray();
            // 是否识别到人脸，返回值
            System.out.println(String.format("Detected %s faces", faces.length));
            int i = 0;
            for (Rect rect : faces) {
                // 循环所有检测到的人脸
                Point x = new Point(rect.x, rect.y);
                Point y = new Point(rect.x + rect.width, rect.y + rect.height);
                // 在image图片上画框，x，y可确定框的位置和大小，new Scalar(0,255,0)是框的颜色，自行调整
                // Imgproc.rectangle(image, x, y, new Scalar(0, 255, 0)); // 保存检测的人脸小图片
                Rect r = new Rect(x, y);
                //System.out.println(r.height + ":" + r.width);
                Mat areaM = new Mat(image, r); // 保存检测的人脸小图片到tmp+序号的jpg文件
                String tmpFilePath = savePath + "tmp" + (i++) + ".png";
                Imgcodecs.imwrite(tmpFilePath, areaM);
                cutFaceSavePosition.add(tmpFilePath);
            }
            // 保存画了方框的图片
            // String filename = savePath + "all.png";
            //Imgcodecs.imwrite(filename, image);
            // 销毁
            image.release();
            return cutFaceSavePosition;
        } else {
            System.out.println("加载分类器失败！请检查文件路径是否正确。");
            return null;
        }
    }

    public static String calculateColorScore(String path) throws Exception {
        // 解决awt报错问题
        System.setProperty("java.awt.headless", "false");
        // System.out.println(System.getProperty("java.library.path"));

        // 加载动态库
        URL url = ClassLoader.getSystemResource("lib/opencv_java343.dll");
        System.load(url.getPath());
        Mat src = imread(path, IMREAD_COLOR);
        if (src.empty()) {
            throw new Exception("image is empty");
        }
        //分割r g b
        List<Mat> dst = new java.util.ArrayList<Mat>(3);
        split(src, dst);
        Mat b = dst.get(0);
        Mat g = dst.get(1);
        Mat r = dst.get(2);
        Mat rg = new Mat();
        subtract(r, g, rg);
        Mat yb = new Mat();
        add(r, g, yb);
        addWeighted(yb, 0.5, yb, 0.0, 0.0, yb);
        subtract(yb, b, yb);
        MatOfDouble rgMeanMat = new MatOfDouble(), rgStdMat = new MatOfDouble(), ybMeanMat = new MatOfDouble(), ybStdMat = new MatOfDouble();
        meanStdDev(rg, rgMeanMat, rgStdMat);
        meanStdDev(yb, ybMeanMat, ybStdMat);
        double rgMean = rgMeanMat.toArray()[0];
        double rgStd = rgMeanMat.toArray()[0];
        double ybMean = ybMeanMat.toArray()[0];
        double ybStd = ybStdMat.toArray()[0];
        double stdRoot = Math.sqrt(rgStd * rgStd + ybStd * ybStd);
        double meanRoot = Math.sqrt(rgMean * rgMean + ybMean * ybMean);
        DecimalFormat df = new DecimalFormat("#.00");
        return String.valueOf(df.format(stdRoot + (0.3 * meanRoot)));
    }

    public static void cutPhotoFace(double lx, double ly, double rx, double ry, String path, String savePath) throws Exception {
        // 解决awt报错问题
        System.setProperty("java.awt.headless", "false");
        // 加载动态库
        URL url = ClassLoader.getSystemResource("lib/opencv_java343.dll");
        System.load(url.getPath());
        Mat src = imread(path, IMREAD_COLOR);
        if (src.empty()) {
            throw new Exception("image is empty");
        }
        Rect rect = new Rect(new Point(lx - 20 >= 0 ? lx - 20 : 0, ly - 20 >= 0 ? ly - 20 : 0), new Point(rx + 20 < src.cols() ? rx + 20 : src.cols(), ry + 20 < src.rows() ? ry + 20 : src.rows()));

        Mat cutImage = cutImage(src, rect);
        Imgcodecs.imwrite(savePath, cutImage);
    }

    public static Mat cutImage(Mat src, Rect rect) {
        //图片裁剪
        Mat src_roi = new Mat(src, rect);
        Mat cutImage = new Mat();
        src_roi.copyTo(cutImage);
        return cutImage;
    }
}
