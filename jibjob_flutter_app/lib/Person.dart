class Person {
  static int nb = 0 ;
  String? name ;
  String ? email ;
  String? password ;
  String ? phone ;
  String ? address ;
  String ? description ;
  String ? image ;

  Person({
    this.name,
    this.email,
    this.password,
    this.phone,
    this.address,
    this.description,
    this.image,
  }) {
    nb++ ;
  }
}

List<Person> Liste_Clients = [];