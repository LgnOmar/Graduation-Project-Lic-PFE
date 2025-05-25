class Listeprix {
  String ? rate ;
  String? name ;
  String ? address ;
  String ? image ;
  final List<Map<String, String>> tarifs;

  Listeprix({
    required this.name,
    required this.address,
    required this.image,
    required this.tarifs,
    required this.rate,
  });

}

List<Listeprix> Pros_Offers = [];