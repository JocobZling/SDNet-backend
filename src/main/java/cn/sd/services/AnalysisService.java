package cn.sd.services;

import cn.sd.entities.vo.AnalysisVo;
import cn.sd.entities.vo.ImageDisplayVo;
import cn.sd.utils.Base64Util;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Objects;
import java.util.UUID;

@Service
public class AnalysisService {
    @Value("${uploadAddr}")
    private String uploadAddr;

    @Value("${histPosition}")
    private String histPosition;

    @Value("${environmentPosition}")
    private String environmentPosition;

    private final FaceService faceService;

    public AnalysisService(FaceService faceService) {
        this.faceService = faceService;
    }

    public AnalysisVo histImage(MultipartFile file) throws Exception {
        AnalysisVo analysisVo = new AnalysisVo();
        assert file != null;
        ImageDisplayVo imageDisplayVo = faceService.encryptedImage(file, 1L);
        // 原图
        String originUrl = imageDisplayVo.getOriginalImagePosition();
        String originPath = uploadAddr + "/" + originUrl.substring(originUrl.lastIndexOf("/") + 1);
        String destPath = upload(uploadAddr, Objects.requireNonNull(file.getOriginalFilename()));
        String[] path = new String[]{originPath, destPath};
        String[] imagePathes = getPythonResult(path);

        assert imagePathes != null;
        analysisVo.setPictureOriginal(Base64Util.encryptToBase64(imagePathes[0]));
        analysisVo.setHistOriginal(Base64Util.encryptToBase64(imagePathes[1]));


        // 分解图1
        originUrl = imageDisplayVo.getPictureOnePosition();
        path[0] = uploadAddr + "/" + originUrl.substring(originUrl.lastIndexOf("/") + 1);
        path[1] = upload(uploadAddr, Objects.requireNonNull(file.getOriginalFilename()));
        imagePathes = getPythonResult(path);
        assert imagePathes != null;
        analysisVo.setPictureOne(Base64Util.encryptToBase64(imagePathes[0]));
        analysisVo.setHistOne(Base64Util.encryptToBase64(imagePathes[1]));
        // 分解图2
        originUrl = imageDisplayVo.getPictureTwoPosition();
        path[0] = uploadAddr + "/" + originUrl.substring(originUrl.lastIndexOf("/") + 1);
        path[1] = upload(uploadAddr, Objects.requireNonNull(file.getOriginalFilename()));
        imagePathes = getPythonResult(path);
        assert imagePathes != null;
        analysisVo.setPictureTwo(Base64Util.encryptToBase64(imagePathes[0]));
        analysisVo.setHistTwo(Base64Util.encryptToBase64(imagePathes[1]));
        return analysisVo;
    }

    private String[] getPythonResult(String[] imagePath) {
        Process proc;
        try {
            //此处的python环境
            String[] arguments = new String[]{environmentPosition, histPosition, imagePath[0], imagePath[1]};
            proc = Runtime.getRuntime().exec(arguments);
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
            System.out.println(imagePath[0]);
            System.out.println(imagePath[1]);
            return imagePath;

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return null;
    }

    private String upload(String path, String fileName) throws Exception {
        // 生成新的文件名
        return path + "/" + UUID.randomUUID().toString().replace("-", "") + fileName.substring(fileName.lastIndexOf("."));
    }



}
