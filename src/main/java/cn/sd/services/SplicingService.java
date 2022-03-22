package cn.sd.services;

import cn.sd.entities.Detection;
import cn.sd.entities.vo.ImageDisplayVo;
import cn.sd.repositories.DetectionRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;
import java.util.UUID;

@Service
public class SplicingService {

    @Value("${uploadAddr}")
    private String uploadAddr;

    @Value("${detectPosition}")
    private String detectPosition;

    @Value("${environmentPosition}")
    private String environmentPosition;

    @Value("${imageUrl}")
    private String imageUrl;

    private final DetectionRepository detectionRepository;

    public SplicingService(DetectionRepository detectionRepository) {
        this.detectionRepository = detectionRepository;
    }

    public ImageDisplayVo encryptedImage(MultipartFile file, Long userId) throws Exception {
        //上传图片
        assert file != null;
        String[] imagePath = upload(file, uploadAddr, Objects.requireNonNull(file.getOriginalFilename()));
        //调用python文件加密图片
        String[] imagePathes = getPythonResult(imagePath);

        Detection detection = new Detection();
        ImageDisplayVo imageDisplayVo = new ImageDisplayVo();
        detection.setUserId(userId);

        assert imagePathes != null;
        detection.setPictureOnePosition(imageUrl + imagePathes[1].split(uploadAddr + "/")[1]);
        detection.setPictureTwoPosition(imageUrl + imagePathes[2].split(uploadAddr + "/")[1]);
        detection.setOriginalImagePosition(imageUrl + imagePathes[0].split(uploadAddr + "/")[1]);
        imageDisplayVo.setOriginalImagePosition(imageUrl + imagePathes[0].split(uploadAddr + "/")[1]);
        imageDisplayVo.setPictureOnePosition(imageUrl + imagePathes[1].split(uploadAddr + "/")[1]);
        imageDisplayVo.setPictureTwoPosition(imageUrl + imagePathes[2].split(uploadAddr + "/")[1]);
        detectionRepository.save(detection);
        return imageDisplayVo;
    }

    private String[] getPythonResult(String[] imagePath) {

        Process proc;
        try {
            //此处的python环境
            String[] arguments = new String[]{environmentPosition, detectPosition, imagePath[0], imagePath[1], imagePath[2]};
            proc = Runtime.getRuntime().exec(arguments);
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
            return imagePath;
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return null;
    }

    private String[] upload(MultipartFile file, String path, String fileName) throws Exception {
        // 生成新的文件名
        String[] realPath = new String[3];

        realPath[0] = path + "/" + UUID.randomUUID().toString().replace("-", "") + fileName.substring(fileName.lastIndexOf("."));
        File dest = new File(realPath[0]);
        // 判断文件父目录是否存在
        if (!dest.getParentFile().exists()) {
            dest.getParentFile().mkdir();
        }
        // 保存文件
        file.transferTo(dest);
        realPath[1] = path + "/" + UUID.randomUUID().toString().replace("-", "") + fileName.substring(fileName.lastIndexOf("."));
        realPath[2] = path + "/" + UUID.randomUUID().toString().replace("-", "") + fileName.substring(fileName.lastIndexOf("."));
        return realPath;
    }
}
