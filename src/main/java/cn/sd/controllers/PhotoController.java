package cn.sd.controllers;

import cn.sd.services.PhotoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.servlet.http.HttpServletRequest;


@RestController
@RequestMapping(value = "/api/photo")
public class PhotoController {

    private final PhotoService photoService;

    @Autowired
    public PhotoController(PhotoService photoService) {
        this.photoService = photoService;
    }

    @PostMapping(value = "/upload/{userId}")
    public ResponseEntity<?> photoUpload(@RequestParam(value = "file") MultipartFile[] files, @PathVariable Long userId, HttpServletRequest request) throws Exception {
        photoService.savePhotoToFile(files, userId);
        return ResponseEntity.status(HttpStatus.OK).build();
    }
    // 按照日期返回 最近的在最前面
    @GetMapping(value = "/days/all/{userId}")
    public ResponseEntity<?> getAllPhoto(@PathVariable Long userId) {
        return ResponseEntity.ok(photoService.findAllDaysPhotosByUserId(userId));
    }

    @GetMapping(value = "/day/{date}/all/{userId}")
    public ResponseEntity<?> getOneDayAllPhoto(@PathVariable String date, @PathVariable Long userId) throws Exception {
        return ResponseEntity.ok(photoService.findOneDaysAllPhotosByUserId(userId, date));
    }

    @GetMapping(value = "/type/{type}/{userId}")
    public ResponseEntity<?> getOneTypeAllPhoto(@PathVariable String type, @PathVariable Long userId) {
        return ResponseEntity.ok(photoService.findOneTypeAllPhotosByUserId(type, userId));
    }

    // 返回来的按日期排序
    @GetMapping(value = "/all/eight/{userId}")
    public ResponseEntity<?> getOneDayEightPhoto(@PathVariable Long userId) {
        return ResponseEntity.ok(photoService.findAllTimesEightPhotoByUserId(userId));
    }

    @GetMapping(value = "/all/type/one/{userId}")
    public ResponseEntity<?> getAllTypeOnePhoto(@PathVariable Long userId) {
        return ResponseEntity.ok(photoService.findAllTypeOnePhotoByUserId(userId));
    }
}
