package cn.sd.controllers;

import cn.sd.entities.User;
import cn.sd.entities.vo.UserEmailAndPassword;
import cn.sd.exceptions.BusinessException;
import cn.sd.services.UserCenterService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;


@RestController
@RequestMapping(value = "/api/users")
public class UserController {

    private final UserCenterService userCenterService;

    // 构造方法
    @Autowired
    public UserController(UserCenterService userCenterService) {
        this.userCenterService = userCenterService;
    }

    @PostMapping("/login")
    public ResponseEntity<HashMap<String, Object>> getUserByEmailAndPassword(@RequestBody UserEmailAndPassword userInfo) throws BusinessException, UnsupportedEncodingException {
        return ResponseEntity.ok(userCenterService.findUserByEmailAndPasswordOrThrow(userInfo.getEmail(), userInfo.getPassword()));
    }

    @PostMapping("/register")
    public ResponseEntity registerUser(@RequestBody User info) throws UnsupportedEncodingException {
        Boolean isUserExits = userCenterService.isUserExits(info.getEmail());
        if (isUserExits) {
            return new ResponseEntity<>("账户已存在", HttpStatus.BAD_REQUEST);
        }
        return ResponseEntity.ok(userCenterService.isRegisterUser(info));

    }

    @PostMapping("/testregist")
    public ResponseEntity testRegister(@RequestBody UserEmailAndPassword userInfo) throws BusinessException, UnsupportedEncodingException {
        System.out.println(userInfo.getEmail());
        Boolean isUserExits = userCenterService.isUserExits(userInfo.getEmail());
        System.out.println(isUserExits);
        if (isUserExits) {
            return ResponseEntity.ok(userCenterService.findUserByEmailAndPasswordOrThrow(userInfo.getEmail(), userInfo.getPassword()));
        }
        return new ResponseEntity<>("用户还未注册", HttpStatus.BAD_REQUEST);
    }

    @PutMapping("/password/{userId}")
    public ResponseEntity updatePassword(@PathVariable Long userId,@RequestBody Map passwordMap){

        try{
            return new ResponseEntity(
                    userCenterService.authUserOldPasswordUpdatePassword(userId,
                            passwordMap.get("oldPassword").toString(),
                            passwordMap.get("newPassword").toString()),
            HttpStatus.NO_CONTENT);
        }catch (BusinessException e) {
            return new ResponseEntity<>("旧密码错误", HttpStatus.BAD_GATEWAY);
        }
    }
    @PutMapping("/profile/{userId}")
    public ResponseEntity updateUserById(@PathVariable Long userId,@RequestBody User user)throws BusinessException,UnsupportedEncodingException{
        Boolean isUserExists = userCenterService.isUserExits(user.getEmail());
        if(!isUserExists){
            return new ResponseEntity(userCenterService.updateUserById(userId,user),HttpStatus.OK);
        }else{
            return new ResponseEntity<>("邮箱重复！", HttpStatus.BAD_REQUEST);
        }
    }

    @GetMapping("")
    public ResponseEntity<String> getUser() {
        return ResponseEntity.ok("okok123321");
    }

}
