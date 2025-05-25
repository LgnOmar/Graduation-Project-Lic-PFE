class Offre {
  String? demande;
  String? phone;
  String? address;
  final List<String?> images ;

  Offre({
    required this.demande,
    required this.phone,
    required this.address,
    required this.images,
  }) ;

  
}

List<Offre> Liste_Offres = [];