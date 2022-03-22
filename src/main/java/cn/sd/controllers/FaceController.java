package cn.sd.controllers;

import cn.sd.exceptions.BusinessException;
import cn.sd.services.FaceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
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

    @GetMapping("/faceDetection/{detectionId}/{userId}")
    public ResponseEntity<?> faceDetect(@PathVariable Long detectionId, @PathVariable Long userId) throws BusinessException {
        return ResponseEntity.ok(faceService.getDetectResult(detectionId));
    }

    @GetMapping("/detectionDetail/{detectionId}")
    public ResponseEntity<?> getDetectDetail(@PathVariable Long detectionId) throws BusinessException {
        return ResponseEntity.ok(faceService.getDetectionDetail(detectionId));
    }
}
