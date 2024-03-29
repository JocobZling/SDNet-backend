package cn.sd.controllers;

import cn.sd.entities.vo.ImageDisplayVo;
import cn.sd.exceptions.BusinessException;
import cn.sd.services.FaceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;


@RestController
@RequestMapping(value = "/api/face")
public class FaceController {

    private final FaceService faceService;

    @Autowired
    public FaceController(FaceService faceService) {
        this.faceService = faceService;
    }

    @PostMapping("/encryptedImage/{userId}")
    public ResponseEntity<?> encryptedImage(@RequestParam(value = "file") MultipartFile file, @PathVariable Long userId, HttpServletRequest request) throws Exception {
        return ResponseEntity.ok(faceService.encryptedImage(file, userId));
    }

    @PostMapping("/encryptedVideo/{userId}")
    public ResponseEntity<ImageDisplayVo> uploadVideo(@RequestParam(value = "file") MultipartFile file, @PathVariable Long userId) throws Exception {
        return ResponseEntity.ok(faceService.uploadVideo(file, userId));
    }

    @GetMapping("/faceDetection/{detectionId}/{type}")
    public ResponseEntity<?> faceDetect(@PathVariable Long detectionId, @PathVariable String type) throws BusinessException {
        return ResponseEntity.ok(faceService.getDetectResult(detectionId, type));
    }

    @GetMapping("/detectionDetail/{detectionId}/{type}")
    public ResponseEntity<?> getDetectDetail(@PathVariable Long detectionId, @PathVariable String type) throws BusinessException {
        return ResponseEntity.ok(faceService.getDetectionDetail(detectionId, type));
    }
}
