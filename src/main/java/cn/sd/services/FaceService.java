package cn.sd.services;

import cn.sd.entities.Detection;
import cn.sd.entities.vo.DetectResultDisplayVo;
import cn.sd.entities.vo.ImageDisplayVo;
import cn.sd.exceptions.BusinessException;
import cn.sd.repositories.DetectionRepository;
import cn.sd.utils.Base64Util;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;

import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

@Service
public class FaceService {

    @Value("${uploadAddr}")
    private String uploadAddr;

    @Value("${detectionPosition}")
    private String detectionPosition;

    @Value("${environmentPosition}")
    private String environmentPosition;

    @Value("${imageUrl}")
    private String imageUrl;

    @Value("${encryptedImagePosition}")
    private String encryptedImagePosition;

    @Value("${videoDetectionPosition}")
    private String videoDetectionPosition;

    @Autowired
    private RedisTemplate redisTemplate;

    private final DetectionRepository detectionRepository;

    public FaceService(DetectionRepository detectionRepository) {
        this.detectionRepository = detectionRepository;
    }

    public ImageDisplayVo uploadVideo(MultipartFile file, Long userId) throws Exception {
        //上传图片
        assert file != null;
        String videoPath = uploadVideo(file, uploadAddr, Objects.requireNonNull(file.getOriginalFilename()));
        Detection detection = new Detection();
        ImageDisplayVo videoDisplayVo = new ImageDisplayVo();

        detection.setUserId(userId);
        assert videoPath != null;
        detection.setOriginalImagePosition(imageUrl + videoPath.split(uploadAddr + "/")[1]);
        videoDisplayVo.setOriginalImagePosition(imageUrl + videoPath.split(uploadAddr + "/")[1]);
        Detection result = detectionRepository.save(detection);
        videoDisplayVo.setDetectionId(result.getId());
        return videoDisplayVo;
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
        Detection result = detectionRepository.save(detection);
        imageDisplayVo.setDetectionId(result.getId());
        return imageDisplayVo;
    }

    private String[] getPythonResult(String[] imagePath) {
        Process proc;
        try {
            //此处的python环境
            String[] arguments = new String[]{environmentPosition, encryptedImagePosition, imagePath[0], imagePath[1], imagePath[2]};
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

    private String uploadVideo(MultipartFile file, String path, String fileName) throws Exception {
        // 生成新的文件名
        String realPath = "";

        realPath = path + "/" + UUID.randomUUID().toString().replace("-", "") + fileName.substring(fileName.lastIndexOf("."));
        File dest = new File(realPath);
        // 判断文件父目录是否存在
        if (!dest.getParentFile().exists()) {
            dest.getParentFile().mkdir();
        }
        // 保存文件
        file.transferTo(dest);
        return realPath;
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
        realPath[1] = path + "/" + UUID.randomUUID().toString().replace("-", "") + ".jpg";
        realPath[2] = path + "/" + UUID.randomUUID().toString().replace("-", "") + ".jpg";
        return realPath;
    }

    public DetectResultDisplayVo getDetectResult(Long detectionId, String type) throws BusinessException {
        String pythonPosition = "";
        if (type.equals("video")) {
            pythonPosition = videoDetectionPosition;
        } else {
            pythonPosition = detectionPosition;
        }
        Detection detection = detectionRepository.findById(detectionId).orElseThrow(() -> new BusinessException("检测失败，请重新上传!"));
        String originalImage = uploadAddr + "/" + detection.getOriginalImagePosition().split("/images/")[1];

        Process proc;
        LinkedList<String> result = new LinkedList<>();
        Date startTime = new Date();
        try {
            //此处的python环境
            String[] arguments = new String[]{environmentPosition, pythonPosition, originalImage, detectionId.toString()};
            proc = Runtime.getRuntime().exec(arguments);
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
                result.add(line);
            }
            in.close();
            proc.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        Date endTime = new Date();
        detection.setStartTime(startTime);
        detection.setEndTime(endTime);
        detection.setResult(result.get(0) + ";" + result.get(1));
        detectionRepository.save(detection);
        DetectResultDisplayVo detectDisplayResult = new DetectResultDisplayVo();
        detectDisplayResult.setResult1(result.get(0));
        detectDisplayResult.setResult2(result.get(1));
        return detectDisplayResult;
    }


    public HashMap<String, Object> getDetectionDetail(Long detectId, String type) throws BusinessException {
        long len = redisTemplate.opsForList().size(detectId.toString());
        if (!redisTemplate.hasKey(detectId.toString())) {
            return new HashMap<String, Object>() {{
                put("textAreaValue", new String[0]);
                put("flag", "go on");
            }};
        }
        if (!redisTemplate.opsForList().index(detectId.toString(), len - 1).equals("end")) {
            return new HashMap<String, Object>() {{
                put("textAreaValue", redisTemplate.opsForList().range(detectId.toString(), 0, -1));
                put("flag", "go on");
            }};
        } else {
            Detection detection = detectionRepository.findById(detectId).orElseThrow(() -> new BusinessException("检测失败，请重新上传加密图片!"));
            String result = detection.getResult();
            if (result != null) {
                String originalBase64 = "video";
                if (!type.equals("video"))
                    originalBase64 = detection.getOriginalImagePosition();
                String finalOriginalBase6 = originalBase64;
                return new HashMap<String, Object>() {{
                    put("originalBase64", finalOriginalBase6);
                    put("textAreaValue", redisTemplate.opsForList().range(detectId.toString(), 0, -1));
                    put("flag", "STOP");
                    put("result", result.split(";"));
                }};
            } else {
                return new HashMap<String, Object>() {{
                    put("textAreaValue", redisTemplate.opsForList().range(detectId.toString(), 0, -1));
                    put("flag", "go on");
                }};
            }
        }
    }
}
