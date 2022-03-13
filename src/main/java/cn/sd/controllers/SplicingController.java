package cn.sd.controllers;

import cn.sd.services.SplicingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;


@RestController
@RequestMapping(value = "/api/splicing")
public class SplicingController {

    private final SplicingService splicingService;

    @Autowired
    public SplicingController(SplicingService splicingService) {
        this.splicingService = splicingService;
    }

    @PostMapping("/encryptedImage/{userId}")
    public ResponseEntity<?> detection(@RequestParam(value = "file") MultipartFile file, @PathVariable Long userId, HttpServletRequest request) throws Exception {
        return ResponseEntity.ok(splicingService.encryptedImage(file, userId));
    }
}
