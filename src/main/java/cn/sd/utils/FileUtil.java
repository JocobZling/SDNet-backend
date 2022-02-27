package cn.sd.utils;

import org.apache.commons.fileupload.FileItem;
import org.apache.commons.fileupload.FileItemFactory;
import org.apache.commons.fileupload.disk.DiskFileItemFactory;
import org.springframework.util.Base64Utils;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.commons.CommonsMultipartFile;
import org.apache.commons.io.FileUtils;

import javax.imageio.ImageIO;
import java.awt.*;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

public class FileUtil {
    public static boolean base64ToFile(String filePath, String base64Data) throws Exception {
        String dataPrix = "";
        String data = "";

        if (base64Data == null || "".equals(base64Data)) {
            return false;
        } else {
            String[] d = base64Data.split("base64,");
            if (d.length == 2) {
                dataPrix = d[0];
                data = d[1];
            } else {
                return false;
            }
        }
        String[] d2 = data.split("\"");

        // 因为BASE64Decoder的jar问题，此处使用spring框架提供的工具包
        byte[] bs = Base64Utils.decodeFromString(d2[0]);
        // 使用apache提供的工具类操作流
        org.apache.commons.io.FileUtils.writeByteArrayToFile(new File(filePath), bs);

        return true;
    }

    public static List<String> saveFile(MultipartFile file, String path) {
        BufferedOutputStream bw = null;
        List<String> pathList = new ArrayList<>();
        try {
            String fileName = file.getOriginalFilename();
            //判断是否有文件且是否为图片文件
            if (fileName != null && !"".equalsIgnoreCase(fileName.trim()) && isImageFile(fileName)) {
                String name = UUID.randomUUID().toString().replaceAll("-", "").substring(0, 10) + getFileType(fileName);
                //创建输出文件对象
                File outFile = new File(path + name);
                //拷贝文件到输出文件对象
                FileUtils.copyInputStreamToFile(file.getInputStream(), outFile);
                pathList.add(name);
                pathList.add(path + name);
                return pathList;
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            try {
                if (bw != null) {
                    bw.close();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return null;
    }

    public static String encryptToBase64(String filePath) {
        if (filePath == null) {
            return null;
        }
        try {
            byte[] b = Files.readAllBytes(Paths.get(filePath));
            return Base64.getEncoder().encodeToString(b);
        } catch (IOException e) {
            e.printStackTrace();
        }

        return null;
    }


    public static MultipartFile fileToMultipartFile(String filePath) {
        File file = new File(filePath);
        FileItem fileItem = createFileItem(file);
        MultipartFile multipartFile = new CommonsMultipartFile(fileItem);
        return multipartFile;
    }

    private static FileItem createFileItem(File file) {
        FileItemFactory factory = new DiskFileItemFactory(16, null);
        FileItem item = factory.createItem("textField", "text/plain", true, file.getName());
        int bytesRead = 0;
        byte[] buffer = new byte[8192];
        try {
            FileInputStream fis = new FileInputStream(file);
            OutputStream os = item.getOutputStream();
            while ((bytesRead = fis.read(buffer, 0, 8192)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            os.close();
            fis.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return item;
    }

    private static Boolean isImageFile(String fileName) {
        String[] img_type = new String[]{".jpg", ".jpeg", ".png", ".gif", ".bmp"};
        if (fileName == null) {
            return false;
        }
        fileName = fileName.toLowerCase();
        for (String type : img_type) {
            if (fileName.endsWith(type)) {
                return true;
            }
        }
        return false;
    }

    private static String getFileType(String fileName) {
        if (fileName != null && fileName.indexOf(".") >= 0) {
            return fileName.substring(fileName.lastIndexOf("."), fileName.length());
        }
        return "";
    }

    public static List<String> getPictureSize(String path) throws IOException {
        File picFile = new File(path);
        if (!picFile.exists() || picFile.isDirectory()) {
            throw new IOException("文件不存在!");
        }
        List<String> imgInfo = new ArrayList<>();
        Image src = ImageIO.read(picFile);
        imgInfo.add(String.valueOf(src.getWidth(null)));
        imgInfo.add(String.valueOf(src.getHeight(null)));
        return imgInfo;
    }

//    public static MultipartFile fileToMultipart(String filePath) {
//        try {
//            // File转换成MutipartFile
//            File file = new File(filePath);
//            FileInputStream inputStream = new FileInputStream(file);
//            MultipartFile multipartFile = new MockMultipartFile(file.getName(), "png", "image/png", inputStream);
//            return multipartFile;
//        } catch (IOException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//            return null;
//        }
//    }
//
//    public static MultipartFile fileToMultipartZip(String filePath) {
//        try {
//            // File转换成MutipartFile
//            File file = new File(filePath);
//            FileInputStream inputStream = new FileInputStream(file);
//            MultipartFile multipartFile = new MockMultipartFile(file.getName(), "zip", "application/x-zip-compressed", inputStream);
//            return multipartFile;
//        } catch (IOException e) {
//            // TODO Auto-generated catch block
//            e.printStackTrace();
//            return null;
//        }
//    }
}
