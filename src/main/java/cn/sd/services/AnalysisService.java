package cn.sd.services;

import cn.sd.entities.vo.AnalysisVo;
import cn.sd.entities.vo.ImageDisplayVo;
import cn.sd.utils.Base64Util;
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
public class AnalysisService {
    @Value("${uploadAddr}")
    private String uploadAddr;

    @Value("${histPosition}")
    private String histPosition;

    @Value("${environmentPosition}")
    private String environmentPosition;

    public AnalysisService() {}

    public AnalysisVo histImage(MultipartFile file) throws Exception {
        AnalysisVo analysisVo = new AnalysisVo();
        assert file != null;

        String[] destPath = upload(file, uploadAddr, Objects.requireNonNull(file.getOriginalFilename()));

        String[] imagePathes = getPythonResult(destPath);

        assert imagePathes != null;
        analysisVo.setPictureOriginal(Base64Util.encryptToBase64(imagePathes[0]));
        analysisVo.setPictureOne(Base64Util.encryptToBase64(imagePathes[1]));
        analysisVo.setPictureTwo(Base64Util.encryptToBase64(imagePathes[2]));
        analysisVo.setHistOriginal(Base64Util.encryptToBase64(imagePathes[3]));
        analysisVo.setHistOne(Base64Util.encryptToBase64(imagePathes[4]));
        analysisVo.setHistTwo(Base64Util.encryptToBase64(imagePathes[5]));

        // cleanCache(imagePathes);

        return analysisVo;
    }

    private String[] getPythonResult(String[] imagePath) {
        Process proc;
        try {
            //此处的python环境
            String[] arguments = new String[]{environmentPosition, histPosition, imagePath[0], imagePath[1],
                    imagePath[2], imagePath[3], imagePath[4], imagePath[5]};
            proc = Runtime.getRuntime().exec(arguments);
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
//            System.out.println(imagePath[0]);
//            System.out.println(imagePath[1]);
            return imagePath;

        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        return null;
    }

    private String[] upload(MultipartFile file, String path, String fileName) throws Exception {
         // 生成新的文件名
        String[] realPath = new String[6];
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
        realPath[3] = path + "/" + UUID.randomUUID().toString().replace("-", "") + fileName.substring(fileName.lastIndexOf("."));
        realPath[4] = path + "/" + UUID.randomUUID().toString().replace("-", "") + fileName.substring(fileName.lastIndexOf("."));
        realPath[5] = path + "/" + UUID.randomUUID().toString().replace("-", "") + fileName.substring(fileName.lastIndexOf("."));
        return realPath;
    }

    private Boolean cleanCache(String[] realpath){
        File cache = new File(realpath[0]);
        cache.delete();
        cache = new File(realpath[1]);
        cache.delete();
        cache = new File(realpath[2]);
        cache.delete();
        cache = new File(realpath[3]);
        cache.delete();
        cache = new File(realpath[4]);
        cache.delete();
        cache = new File(realpath[5]);
        return cache.delete();
    }
}
