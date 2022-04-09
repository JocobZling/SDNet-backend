package cn.sd.config;
import java.io.FileWriter;
import java.io.IOException;

public class AlipayConfig {
    // 应用ID,我的APPID，收款账号既是您的APPID对应支付宝账号
    public static String app_id = "2021000119630880";

    // 商户私钥，我的PKCS8格式RSA2私钥
    public static String merchant_private_key = "MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCbxgRGD4BVgJRKJK3P+/mgNJ+B22+ZIfKMiqw1ykSNkUdR06/KaR6IJlhy1xnmOTulXtuV/t83OvRvm9+DZZkhZOer/t8oIxRdZaxXNsrOXhWuN8T0f78JyNWByuzp2s7anquLyjxYcNiZCbehgRTtRUItCw6/QK9nSH6CQRXpoZoJ3Ih9OdBgWWlFNZ9OOF+0QCtzPtHCx4pIQhVLjjOyhO3Bq4NSMLaXKKL9v1CsuVYGWaIYiasOSQWPIwRWQpa99hmXKCf0HVr7UFyY2pDH4MbFcm+jGM8OrrtKJCy/DvP3dVifkh6jwQcNGdsaMy1inMuRV2pZ1y8Ulsx2RQ7lAgMBAAECggEAUEng3AeTWv6Z8Bk1qTSztjpJWJI01BXmQUAcgOEPMnSF8QwN2GMUBJLXFeH+jGZFcvgQcCBdBseRQ5Fww/Xy1n5+1mzYsoRaM8V67bzlDTb7L7+4+Vh6q91jBrIEKIjKA26y9qs3LVKEoWSbzFeO5LjE4hiH8lWx0qdP5W0MkU0TvnrTScyrqwhMfGggXrkego/gEpmPuugSd+UVKjm9eCkz2PENXsm8kAohbyleL+vNGn0ci/KDdArvQAjiRd0jdiMGLaWF0FSEKvQwqE/L7sJAnc/Re7yBQYruQlzbbZX0/AQHA4VXAByH8Hs4Zje8GXXIlGXZLNWfX/wvXoWKoQKBgQDu6l2000SF6XlbnUor6SXLlG42fnL+FHdNkrUqBNa5uH/3LxWmNJzXzkfXAJVZuIIeJWK1cexfLSNABBM4r7dXQpHe0lzCxf2gSElbSnIdnu5LbUNJpEnRuG6aClhqf8DKubbiI/lSd9FImMZaKi3QgI8Uys31mio2OAkvkQfsPQKBgQCm6aMVvsT++6dgR+af9WO4wSMJSUVXQC6AatQBrSxq7whlcJ5Zd7XAeMBJzs6Cmp7m5BAGU7IaukIovZ42K7SnxxGHWJx44NWJsxunAKmcAyqF9EGOBgnoAKnDTy/1P5cjr6yRwRkjb8bZW8zmVj+n07kiu3vmqopQMY0j61kPyQKBgBL/KaJGJi2rqc2iDRK+wBWW4MiOd2BCilXts1/68IPl6q7h3IvRpSL3sPikgDuZEyeIe3fH7i6HLUVazZh1SFO05FpeSafnDEqhVMvgeIBocbQnpPEHEiON0aWH6w3UyKSV7be/eoSXKFnoY+Upc2kfv8C8e3Hq5y5WeEyC8KlVAoGASbfsyIDeHsPk07hN2p4VshbTM5nB31xAbYVBJ5BlisER00OfHGKryltKig6l1LcXsWBO14yFoBpdeALvant5KZTZYA655QVgKpYJg4PeZUJcuJb9kQ2HuIj6cYF66/Vu+lrve9xuDuGiUIOkBqNXL/dp0DY/7cuqlOqdrJyYVEkCgYA0KuIZuEb06VKdghPWhGT48NFbzmuuzjUabXcMnv9qDjXQPlg0/+VlcfN0IYp4uTZPdBf9dv4MavfbzbYdjhYCZ45s1fT9I/ORMmQK2NiiGruTeomeQHUIzpPLR96WTWGIGM2LeWsqZgpwrU8bH1lnJGCIzbJtljSopkDluKsOGA==";

    // 支付宝公钥,查看地址：https://openhome.alipay.com/platform/keyManage.htm 对应APPID下的支付宝公钥。
    public static String alipay_public_key = "MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAiWMcjHmMtk3cJsbLvY9GWp53HK18ZyZNlK87zZ0mVKvQfA93QcVao/qJbrsxS4AhgokO9RFARyQj2HZsatU5EDoT2QH9XLr3ggU/I4+54DIo/AXSRFSFEYhBKaOcQujrpkT7dNSXTJtC4LRbU1Pppe+8cfLZhW2evdCo4j/XqAmfc522LEksiBrT4Xh57sI2Hgai0zXOlFGdfUE0nSli9YBkMUlEa/1xEteIerDMdDU9R8l8K0nPKaAXZJBOfY9UDxvd+/8YnAuY7+lRVHwULv2Lmpm/+DYdoRoR9efPkJ+zIWQqD2FPZ9nrPpItMz5hjh8bOSy5AJf4x9HP3CJWVQIDAQAB";

    // 服务器异步通知页面路径  需http://格式的完整路径，不能加?id=123这类自定义参数，必须外网可以正常访问
    public static String notify_url = "http://22d858i464.51mypc.cn/html/doNotifyUrl";

    // 页面跳转同步通知页面路径 需http://格式的完整路径，不能加?id=123这类自定义参数，必须外网可以正常访问
    public static String return_url = "http://22d858i464.51mypc.cn/html/doReturnUrl";

    // 签名方式
    public static String sign_type = "RSA2";

    // 字符编码格式
    public static String charset = "utf-8";

    // 支付宝网关
    public static String gatewayUrl ="https://openapi.alipaydev.com/gateway.do";

    // 支付宝网关
    public static String log_path = "C:\\";

    /**
     * 写日志，方便测试（看网站需求，也可以改成把记录存入数据库）
     * @param sWord 要写入日志里的文本内容
     */
    public static void logResult(String sWord) {
        FileWriter writer = null;
        try {
            writer = new FileWriter(log_path + "alipay_log_" + System.currentTimeMillis()+".txt");
            writer.write(sWord);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}

