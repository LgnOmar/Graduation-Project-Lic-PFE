class Pro{
  static int nb = 0 ;
  String? name ;
  String ? email ;
  String? password ;
  String ? phone ;
  String ? address ;
  String ? description ;
  String ? image ;
  Pro({
    this.name,
    this.email,
    this.password,
    this.phone,
    this.address,
    this.description,
    this.image,
    List<String?> ? images,
    List<String?> ? Services,
    List<Map<String, String>> ? tarifs ,

  })    {
          nb++ ;
        }

          List<String?>  images = [] ;
          List<String?> Services = [];
          List<Map<String, String>> tarifs = [] ;
          
}

List<Pro> Liste_Pros = [];